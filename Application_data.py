import requests
from requests.structures import CaseInsensitiveDict 
import json
import time
import pandas as pd
import os
import networkx as nx
import plotly.graph_objects as go
import colorsys
import seaborn as sns
from collections import defaultdict #for line style
import dash
import networkx as nx
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = html.H2("Initializing...")

class AirtableVisualizer():
    '''Explain Class'''
    def __init__(self, config):
        self.airtable_token = config.get("airtable_token")
#region airtable (pull from db)
    def pull_table_api(self, table_id, data = None, offset=None, max_retries=5):
        '''returns all data from a particular table (with pagination)'''

        if data is None:
            data = []

        headers = {
            'Authorization': f'Bearer {self.airtable_token}',
        }
        params = {
            'view': 'Grid view',
        }
        if offset:
            params['offset'] = offset

        retry_count = 0
        while retry_count < max_retries:
            response = requests.get(f"https://api.airtable.com/v0/appY3Ht2ufuvQ0f18/{table_id}", params=params, headers=headers)

            if response.status_code == 200:
                resp_dict = json.loads(response.content.decode('utf-8'))
                data.extend(resp_dict.get('records', []))
                if resp_dict.get('offset'):
                    return self.pull_table_api(table_id, data, resp_dict.get('offset'))
                else:
                    return data
            elif response.status_code == 429:  # Rate limit exceeded
                retry_count += 1
                time.sleep(2)  # Exponential backoff
            else:
                raise Exception(f'Failed to retrieve data: {response.content}')       
    def list_to_first_str(self, record):
        '''used to clean relational data b/c when it is converted from ID to str, str is in a list, but final table should not be lists'''
        cleaned_record = {}  # Create an empty dictionary to hold the cleaned data
        for key, value in record.items():  # Loop through each key-value pair in the record
            # If the value is a list, replace it with its first element, provided the list isn't empty
            if isinstance(value, list) and value:  
                cleaned_record[key] = value[0]
            else:
                cleaned_record[key] = value  # If the value isn't a list, just copy it over unchanged
        return cleaned_record
    def generate_pns_table_ids(self, primary_table_name):
        '''generates the argument for the next formual by providing primary and secondary table ids'''
        headers = {
        'Authorization': f'Bearer {self.airtable_token}',
        }

        baseId='appY3Ht2ufuvQ0f18'
        response = requests.get(f'https://api.airtable.com/v0/meta/bases/{baseId}/tables', headers=headers)
        resp_dict = json.loads(response.content.decode('utf-8'))
        s_id_list = []
        for table in resp_dict.get('tables'):
            if table.get('name') == primary_table_name:
                p_id = table.get('id')
            else:
                s_id_list.append(table.get('id'))

        return p_id, s_id_list
    def completely_populate_one_table(self, primary_table_id, list_of_secondary_table_ids):
        '''used to populate a relational tab on airtable by extracting data from all tables. designed for relational connections just to primary column that is always called "Name" by convention'''

        primary_records = self.pull_table_api(primary_table_id)

        # make a list of all secondary records
        secondary_records = []
        for id in list_of_secondary_table_ids:
            for record in self.pull_table_api(id):
                secondary_records.append(record) 

        # Make a dictionary of all ids and their corresponding values
        id_name_dict = {}
        for s_record in secondary_records:
            id = s_record.get('id')
            name = s_record.get('fields').get('Name')
            id_name_dict[id] = name

        # update primary record with relational values using the dict
        for p_record in primary_records:
            for key, value in p_record.get('fields').items():
                if isinstance(value, list):
                    value = [id_name_dict.get(item) if id_name_dict.get(item) else item for item in value]
                    p_record['fields'][key] = value  # update the original dictionary
                elif isinstance(value, str) and id_name_dict.get(value):
                    p_record['fields'][key] = id_name_dict[id]

        # the resulting list of dictionaries has an asymetrical structure b/c the rest of the fields of the row are kept in a value associated with "fields"
        # I'll need to flatten that to make it ready for converting to pandas df
        partiaully_clean_records = [
        {**{'id': item['id'], 'createdTime': item['createdTime']}, **item['fields']}
        for item in primary_records
        ]
        # all relational data comes as a list, we want to repalce lists with the first value
        clean_records = [self.list_to_first_str(record) for record in partiaully_clean_records]


        return clean_records
    def generate_main_df(self, main_table_name):
        '''generates main df (as input), and rest is secondary (secondary is only used to decode any relational ids in data)
        input should be string that represents Table Name in Airtable'''
        primary_table_id, list_of_secondary_table_ids = self.generate_pns_table_ids(main_table_name)
        records= self.completely_populate_one_table(primary_table_id,list_of_secondary_table_ids)

        return pd.DataFrame(records)
    def generate_dict_from_table(self, table, key_field, value_field, value_field2= ""):
        '''used to make dictionaries like name/biz function dict that can decide color/size of nodes/edges
        value_field2 handles returning a list with both value fields (useful for holding both nodes of the connection)'''
        primary_table_id, list_of_secondary_table_ids = self.generate_pns_table_ids(table)
        records= self.completely_populate_one_table(primary_table_id,list_of_secondary_table_ids)        

        table_df = pd.DataFrame(records)
        if value_field2 == "":
            return {key: value for key, value in zip(table_df[key_field], table_df[value_field])}   
        else:
            return {key: [value1, value2] for key, value1, value2 in zip(table_df[key_field], table_df[value_field], table_df[value_field2])} 
    def generate_duplicate_reference(self):
        '''collects data on duplicate values because they will be handled differently on the group (will have multiple lines, one for each integration)
        I use tuple(sorted()) to make sure its a duplicate even if the to and from are in different orders'''
        complete_tofrom_list = [tuple(sorted(self.integration_name_tofrom_dict[key])) for key in self.integration_name_tofrom_dict]
        
        duplicate_reference = {}
        loop_history = []
        for key in self.integration_name_tofrom_dict:
            current_tofrom =  tuple(sorted(self.integration_name_tofrom_dict[key]))
            dup_index = loop_history.count(current_tofrom)
            loop_history.append(current_tofrom)
            dup_count = complete_tofrom_list.count(current_tofrom)
            if dup_count != 1:
                duplicate_reference[key] = (current_tofrom, dup_count, dup_index)

        return duplicate_reference
    def grab_data(self):
        '''grabs data and creates a set of objects that will help translate into components of the visual down below. 
        b/c bizfunc coordinates the color, we need those, and then conntectiontype will coordinate the lines so we need those, etc...
        returns self.df because that is assumed to be the graph inputs. This allows the graph to rerender with different inputs b/c the graphing equation requires inputs, instead of just assuming the input is self.df and there fore cannot be reconfigured (with slicers)'''
        self.graph_inputs = self.generate_main_df("Application Connections")
        self.dept_df = self.generate_main_df("Department")
        self.integration_name_datatype=self.generate_dict_from_table('Application Connections', "Name", "Data Types")
        self.application_datatype=self.generate_dict_from_table('Applications', "Name", "Data Types")
        self.datatype_sensitivity=self.generate_dict_from_table('Data Types', "Name", "Impact Criteria")
        self.app_dept_dict = self.generate_dict_from_table('Applications', "Name", "Department Driver")
        self.integration_name_bizfunc_dict = self.generate_dict_from_table('Application Connections', "Name", "Business Function")
        self.integration_name_department_dict = self.generate_dict_from_table('Application Connections', "Name", "Department Driver")
        self.integration_name_connecttype_dict=self.generate_dict_from_table('Application Connections', "Name", "Connection Type")
        self.integration_name_tofrom_dict= self.generate_dict_from_table('Application Connections', "Name", 'Application Data From', "Application Data To")
        self.name_description_dict = self.generate_dict_from_table('Application Connections', "Name", "Further Comments")
        self.name_description_dict.update(self.generate_dict_from_table('Applications', "Name", "Function"))
        self.name_department_dict = self.generate_dict_from_table('Application Connections', "Name", "Department Driver")
        self.name_department_dict.update(self.generate_dict_from_table('Applications', "Name", "Department Driver"))
        self.name_datatype_dict = self.generate_dict_from_table('Application Connections', "Name", "Data Types")
        self.name_datatype_dict.update(self.generate_dict_from_table('Applications', "Name", "Data Types"))
        self.duplicate_reference=self.generate_duplicate_reference()
        return self.graph_inputs
#endregion
#region plotly/networkx (arrange network visual)
    #region helpers
    def find_disconnected_clusters(self, G):
        """
        Solves the issues of diconnected clusters overlaying over each other and adding noise to the visual

        finds subgraphs, and creates report of which nodes need to connect to make everything part of one graph
        """
        # Find all connected components
        components = list(nx.connected_components(G))
        if len(components) <= 1:
            # No need to connect clusters if there's only one
            return []

        self.new_connections = []  # List to hold the newly connected node pairs
        for i in range(len(components) - 1):
            # Select a node from the current component and the next
            node_from_current = list(components[i])[0]
            node_from_next = list(components[i + 1])[0]
            # Record the new connection
            self.new_connections.append((node_from_current, node_from_next))
        
        return self.new_connections
    def handle_disconnected_clusters(self, G):
        '''looks for disconeccted clusters, reccomends connections for nodes, adds this to DF with business function CONNECT CLUSTERS'''
        # Sample data for `new_connections`
        new_connections = self.find_disconnected_clusters(G)

        # Define a template for the new rows
        new_row_template = {
            'id': "",
            'createdTime': "",
            'Business Function': "CONNECT CLUSTERS",
            'Further Comments': "CONNECT CLUSTERS",
            'Connection Type': "CONNECT CLUSTERS",
            'Department Driver': "CONNECT CLUSTERS",
            'Application Data To': "",
            'Name': "",
            'Application Data From': "",
            'Status': "Active",
            'Direction': "Bidirectional"
        }

        # Iterate through each connection pair and append a new row to `self.df`
        for i, connection in enumerate(new_connections):
            new_row = new_row_template.copy()
            new_row['Application Data From'] = connection[0]
            new_row['Application Data To'] = connection[1]
            new_row['Name'] = f"CONNECT CLUSTERS {i}" 
        
            # Turn the new row into a DataFrame before concatenation
            new_row_df = pd.DataFrame([new_row])
        
            # Concatenate the new DataFrame with the existing one
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        return new_connections  
    def remove_invisble_edges(self, G, new_connetions):
        '''Connections were fabricated to produce clean pos (positions) on the graph. Once the positions are generated, we want to delete the new connections, as they are not based on the input'''
        for connection in new_connetions:
            if G.has_edge(connection[0], connection[1]):
                pass
                # G.remove_edge(connection[0], connection[1])
    def find_keys_for_edge_pair(self, dictionary, edge_pair):
        '''explain'''
        result_keys = []
        for key, value in dictionary.items():
            # Check if the set of values in the dictionary entry matches the set of items in edge_pair
            if set(value) == set(edge_pair):
                result_keys.append(key)
        return result_keys
    def wrap_text(self, input):
        '''wraps text but replacing spaces with line breaks, making every new space put text on new line'''
        text=str(input)
        return text.replace(' ', '<br>')
    def return_colormap(self):
        '''explain'''
        dept = self.dept_df['Name'].tolist()


        # Determine the number of colors needed
        n_colors = max(len(dept), 20)

        colors_per_palette = n_colors // 3
        remaining_colors = n_colors % 3  # To handle the case where n_colors is not divisible by 3
        color_palette = (
            sns.color_palette("pastel", colors_per_palette) +
            sns.color_palette("Set2", colors_per_palette) +
            sns.color_palette("Set3", colors_per_palette + remaining_colors)  # Add the remaining colors to one of the palettes
        )

        color_map = {
            param: color_palette[i]
            for i, param in enumerate(dept)
        }
        return color_map
    def get_color(self, param):
        '''returns a color per biz func'''
        if param is None:
            return 'rgba(200,200,200,1)'  # Default color for nodes with no matching data
        self.color_map = self.return_colormap()
        color = self.color_map[param]
        return f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
    def get_line_style(self,integration):
        '''maps a particular line style per connection'''
        line_style_map = {
            'Prebuilt Connector': {'color': '#8de5a1', 'line': 'solid'},
            'CLI': {'color': '#8de5a1', 'line': 'solid'},
            'IPaas': {'color': '#888', 'line': 'solid'},
            'VSC Uploads': {'color': '#888', 'line': 'dash'},
            'Manual': {'color': '#ff9f9b', 'line': 'dash'},
            'Home-grown API': {'color': '#888', 'line': 'solid'},
        }

        # this needs to be coded differently!
        connection_type = self.integration_name_connecttype_dict[integration]
        return line_style_map.get(connection_type, {'color': '#888', 'line': 'solid'})
    def handle_line_style(self, G):
        '''groups the line styles'''
        grouped_integration = defaultdict(list)
        for edge_pair in G.edges():
            for integration in self.find_keys_for_edge_pair(self.integration_name_tofrom_dict, edge_pair):
                style = self.get_line_style(integration)
                grouped_integration[(style['color'], style['line'])].append(integration)
        return grouped_integration
    def handle_mnode_position(self, input, integration, x0, x1, y0, y1):
        '''the nodes in the middle of the edge (mnode) it handles multiple integrations per node set by placing many 'mnodes' or middle edge nodes, one per integration
        input needs to be 'y position' or 'x position' '''
        # No need to calculate segment length for single mnode, 
        # as it should be positioned at the midpoint
        midpoint_x = (x0 + x1) / 2
        midpoint_y = (y0 + y1) / 2

        mnode_x_position = midpoint_x
        mnode_y_position = midpoint_y

        mnode_info = self.duplicate_reference.get(integration, (0, 1, 0))
        mnode_total = mnode_info[1]
        mnode_index = mnode_info[2] + .5
        if mnode_total > 1:
            length_x = x1 - x0
            length_y = y1 - y0
            segment_length_x = length_x / (mnode_total + 2)
            segment_length_y = length_y / (mnode_total + 2)
            mnode_x_position = midpoint_x + (mnode_index - mnode_total / 2) * segment_length_x
            mnode_y_position = midpoint_y + (mnode_index - mnode_total / 2) * segment_length_y

        if input == "y position":
            return [mnode_y_position]
        elif input == "x position":
            return [mnode_x_position]
        else:
            return "RECONSIDER  INPUT VARIABLE"
    def handle_node_boarder(self, G):
        '''groups the nodes by the three cases (all sensative data, all non sensative data, mixed) and adds the appropriate styling to the groups'''
        sensitivity_map = defaultdict(list)

        for node in G.nodes():
            # Retrieves the data type for the node, ensuring it's a list
            dt_list = self.name_datatype_dict.get(node[:len(node)], [])
            dt_list = dt_list if isinstance(dt_list, list) else [dt_list]       

            # Creating a set of sensitivities to avoid duplicates
            sensitivity_agg = {self.datatype_sensitivity.get(dt) for dt in dt_list}      

            # Determine the status based on the aggregated sensitivities
            if sensitivity_agg == {'Sensitive'}:
                status = 'sensitive'
            elif sensitivity_agg == {'Non-Sensitive'}:
                status = 'non-sensitive'
            else:
                status = 'mixed'        

            # Append the node to the appropriate category in sensitivity_map
            sensitivity_map[status].append(node)        

            # print(node, ":  ", list(sensitivity_agg), status)

        # Returns the sensitivity map
        return sensitivity_map  
    def integration_in_edges(self, integration_pair, edge_view):
        # Create both possible tuples for the integration pair
        forward_edge = tuple(integration_pair)
        reverse_edge = tuple(reversed(integration_pair))
        if forward_edge in edge_view or reverse_edge in edge_view:
            return True
        return False
    def handle_mnode_boarder(self, G):
        '''groups the nodes by the three cases (all sensative data, all non sensative data, mixed) and adds the appropriate styling to the groups'''
        sensitivity_map = defaultdict(list)

        for integration in self.integration_name_datatype:
            # making sure this cross references G.nodes which holds the filtered nodes
            if self.integration_in_edges(self.integration_name_tofrom_dict[integration], G.edges): 
                # Retrieves the data type for the integration, ensuring it's a list
                dt_list = self.name_datatype_dict.get(integration, [])
                dt_list = dt_list if isinstance(dt_list, list) else [dt_list]       

                # Creating a set of sensitivities to avoid duplicates
                sensitivity_agg = {self.datatype_sensitivity.get(dt) for dt in dt_list}      

                # Determine the status based on the aggregated sensitivities
                if sensitivity_agg == {'Sensitive'}:
                    status = 'sensitive'
                elif sensitivity_agg == {'Non-Sensitive'}:
                    status = 'non-sensitive'
                else:
                    status = 'mixed'        

                # Append the integration to the appropriate category in sensitivity_map
                sensitivity_map[status].append(integration)        

                # print(integration, ":  ", list(sensitivity_agg), status)

        # Returns the sensitivity map
        return sensitivity_map            
    
    #endregion
    def load_data_into_visual(self, graph_inputs):
        '''first step to plotly, data should be df'''
        # should be subset of 1st or second column, not a column w specific name!
        self.df = graph_inputs.dropna(subset=['Application Data To'])
        initial_G = nx.from_pandas_edgelist(self.df, 'Application Data From', 'Application Data To')
        self.new_connections = self.handle_disconnected_clusters(initial_G)
        self.G = nx.from_pandas_edgelist(self.df, 'Application Data From', 'Application Data To')
        pos = nx.kamada_kawai_layout(self.G)
        return self.G, pos
    def handle_edges(self, G, pos):
        '''this handles edges generation'''

        grouped_integration = self.handle_line_style(G)
        traces = []

        for (color, line), integrations in grouped_integration.items():
            edge_x, edge_y = [], []
            for integration in integrations:
                edges = self.integration_name_tofrom_dict[integration]

                x0, y0 = pos[edges[0]]
                x1, y1 = pos[edges[1]]

                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color=color, dash=line),
                hoverinfo='text',
                hovertext="doesn't work",
                mode='lines'
            )
            traces.append(edge_trace)
        return traces
    def handle_mnodes(self, G, pos):
        '''handles mnodes (middle of edge nodes)'''
        sensitivity_formatting = {
            'sensitive': dict(color='rgba(255, 0, 0,.7)', width=1), 
            'mixed':dict(color='rgba(255, 0, 0,.7)', width=1), 
            'non-sensitive':dict(color='rgba(128,128,128,.4)', width=0.5)
        }
        grouped_integrations = self.handle_mnode_border(G)
        traces = []
        for sensitivity in grouped_integrations:
            mnode_x, mnode_y, mnode_txt, mnode_colors, connection_name = [], [], [], [], []
            for integration in grouped_integrations[sensitivity]:
                # to grab the position of the edge, that the mnode will go on
                edges = self.integration_name_tofrom_dict[integration]
                x0, y0 = pos[edges[0]]
                x1, y1 = pos[edges[1]]

                mnode_x.extend(self.handle_mnode_position('x position', integration, x0, x1, y0, y1))
                mnode_y.extend(self.handle_mnode_position('y position', integration, x0, x1, y0, y1))
                integration_dept = self.integration_name_department_dict.get(integration)
                mnode_colors.append(self.get_color(integration_dept))
                mnode_txt.extend([f'Int: {integration}'])
                connection_name.append(integration)
            mnode_trace = go.Scatter(
                x=mnode_x,
                y=mnode_y,
                customdata=connection_name,
                mode="markers",
                showlegend=False,
                hovertemplate="%{hovertext}<extra></extra>",
                hovertext=mnode_txt,
                marker=dict(
                    color=mnode_colors,  # Using the generated color list
                    opacity=0.6,
                    line=sensitivity_formatting[sensitivity]
                )
            )
            traces.append(mnode_trace)

        return traces
    def handle_nodes(self, G, pos):
        '''handles the nodes. First groups the nodes by data sensitivity, then comutes each nodes location, color, and finally creates a formatted trace'''
        grouped_nodes = self.handle_node_border(G)
        traces = []
        self.annotations = []

        for sensitivity in grouped_nodes:
            node_x, node_y, node_text_wrapped, app_name, node_hovertext, node_colors=[],[],[],[],[],[]
            for node in grouped_nodes[sensitivity]:
                node_x.append(pos[node][0])
                node_y.append(pos[node][1])
                node_hovertext.append(f'App: {node}')
                app_name.append(node)
                node_colors.append(self.get_color(self.app_dept_dict.get(node)))
                node_text_wrapped.append(self.wrap_text(node))
            
            outer_border_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    line=self.sensitivity_formatting[sensitivity],
                    size=37,  # Adjust size to be slightly larger than main markers
                ),
                hoverinfo='none',
                # hovertext=node_hovertext,
                customdata=app_name, 
                showlegend=False
            )
            traces.append(outer_border_trace)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                customdata=app_name,
                textposition='middle center',  # Centers text in circles
                hoverinfo='text',
                hovertext=node_hovertext,
                name = sensitivity,
                marker=dict(
                    color=node_colors,
                    size=34,
                    line=dict(color='rgba(229,236,246,255)', width=2),  # Replace with your desired outer border color
                )
            )
            
            traces.append(node_trace)


            node_labels = [
                dict(
                    x=x, y=y,
                xref='x', yref='y',
                    text=text,
                    showarrow=False,
                    font=dict(size=7),
                    align='center'
                )
                for x, y, text in zip(node_x, node_y, node_text_wrapped)
            ]
            self.annotations.extend(node_labels)
        return traces
 
    def generate_viz(self, data):
        '''explain'''
        fig = go.Figure(data=data, 
                        layout=go.Layout(
                            # title='<br>Network Graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                        hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=2),
                            annotations=self.annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=900,  # Example width
                            height=900  # Example height, equal to width to make it square
                        ))
        return fig 
    def handle_visual(self, graph_inputs):
        G, pos = self.load_data_into_visual(graph_inputs)
        # nodes will be colored by department
        self.node_color_param = self.dept_df['Name'].tolist()
        self.sensitivity_formatting = {
            'sensitive': dict(color='rgba(255, 0, 0,1)', width=1), 
            'mixed':dict(color='rgba(255, 0, 0,1)', width=1), 
            'non-sensitive':dict(color='rgba(128,128,128,.7)', width=0.5)
        }
        node_traces = self.handle_nodes(G, pos)
        mnode_traces = self.handle_mnodes(G, pos)
        edge_traces = self.handle_edges(G, pos)
        fig = self.generate_viz(edge_traces + mnode_traces + node_traces)
        return fig
#endregion
#region dash (configure dynamic frontend)
    #region dash helpers
    def separate_list_keep_str(self, value):
        '''if str, return str, if list, return comma seperated str'''
        if isinstance(value, list):
            # Convert list to comma-separated string
            value = ', '.join(value)
        return value
    def create_dept_slicer_options(self, options_list):
        '''builds the slicer label & values'''
        options = []
        # Where is best place to add all?
        for option in options_list:
            color_set = self.color_map[option]
            complete_color = f'rgb({int(color_set[0]*255)}, {int(color_set[1]*255)}, {int(color_set[2]*255)})'
            bullet = html.Span('•', style={'fontSize': '20px', 'fontWeight': 'bold', 'color': complete_color})

            label_text = option
            if option == "Information Technology":
                label_text = "IT"
            elif option == "Digital Construction Team":
                label_text ="DCT"
            elif option == "Management":
                label_text = "MGMT"
            elif option == "Human Resources":
                label_text = "HR"
            elif option == "Project Management":
                label_text = "BUILD" 
            elif option == "All":
                label_text=  "General-Use"

            # Text style set to normal weight and black color
            text_style = {'fontWeight': 'normal', 'color': 'black'}

            label = html.Div(children=[bullet, ' ', html.Span(label_text.upper(), style=text_style)], 
                             style={'display': 'inline-flex', 'alignItems': 'center', 'whiteSpace': 'nowrap'})
            options.append({'label': label, 'value': option}) 
        return options
    def create_connection_slicer_options(self):
        options_raw = [{'label': "Proprietary", 'value': "Proprietary"},{'label': "Manual", 'value': "Manual"}, {'label': "Purchased", 'value': "Purchased"}]
        options = []
        for option in options_raw:
            label_text = option['label']
            if label_text == "Manual":
                color = 'red'
                line = html.Span('--', style={'fontSize': '20px', 'fontWeight': 'bold', 'color': color})
            elif label_text == "Purchased":
                color = 'green'
                line = html.Span('—', style={'fontSize': '20px', 'fontWeight': 'bold', 'color': color})
            else:
                color = 'black'
                line = html.Span('—', style={'fontSize': '20px', 'fontWeight': 'bold', 'color': color})

            text_style = {'fontWeight': 'normal', 'color': 'black'}
            label = html.Div(children=[line, '  ', html.Span(label_text.upper(), style=text_style)], 
                             style={'display': 'inline-flex', 'alignItems': 'center', 'whiteSpace': 'nowrap'})
            options.append({'label': label, 'value': option['value']}) 
        return options
    def dash_wrap_paragraph(self, text):
        '''dash wrap text is different than the plotly wrap text, which was for node titles. this is for descriptions'''

        # Wrap function
        def wrap_line(s, max_len):
            if len(s) <= max_len:
                return s, ""
            break_point = s.rfind(' ', 0, max_len)
            if break_point == -1:
                return s, ""  # Return the whole string if there's no space
            return s[:break_point], s[break_point+1:]   

        # Split for the first line
        line1, rest = wrap_line(text, 58)   

        # Split for the second line
        line2, rest = wrap_line(rest, 68)   

        # Split for the third line
        line3, rest = wrap_line(rest, 65)
        if rest:
            line3 += "..."  # If there's more content, add "..."    

        return list(filter(None, [line1, line2, line3]))
    def is_sensitive_data_type(self, data_type):
        if isinstance(data_type, list):
            # If data_type is a list, check if any of the types in the list is 'Sensitive'
            return any(self.datatype_sensitivity.get(dt) == 'Sensitive' for dt in data_type)
        else:
            # If data_type is not a list, proceed as before
            return self.datatype_sensitivity.get(data_type) == 'Sensitive'
    def filter_apps_bydept(self, selected_app_departments, current_inputs):
        '''has to filter the apps by looking at the integrations and making sure integrations only stay if both apps are in selected departments'''
        filtered_apps = []
        # ! Arbitrary rule that when under four picks, remove the alls
        for dept in selected_app_departments:
            for app in self.app_dept_dict:
                if self.app_dept_dict[app] == dept:
                    filtered_apps.append(app)

        mask_to = current_inputs['Application Data To'].isin(filtered_apps)
        mask_from = current_inputs['Application Data From'].isin(filtered_apps)
        combined_mask = mask_to & mask_from
        filtered_df = current_inputs[combined_mask]
    
        return filtered_df
    def generic_filter(self, selected_options, selected_options_dict, filter_column, current_inputs):
        '''filters df with options from column. selected_options_dict translates the selected options into backend options'''
        true_selected_options = []
        for option in selected_options:
            true_selected_options.extend(selected_options_dict[option])

        filtered_df = current_inputs[current_inputs[filter_column].isin(true_selected_options )]
        return filtered_df
    #endregion
    def run_dash(self, app, fig):
        """
        Configure and run the Dash application with the provided network graph and legend.
        This function sets up the layout and callbacks of the Dash app.

        Parameters:
            app (dash.Dash): The Dash application instance.
            fig (go.Figure): The Plotly graph object for the network graph.
            legend_fig (go.Figure): The Plotly graph object for the legend.
        """

        options_list = [value for value in list(set(self.integration_name_department_dict.values())) if str(type(value)) == "<class 'str'>"]
        options_list.append('All')
        app.layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.H2("IT Application Integration Map", className="text-center"),
                        width=12,
                        style = {
                            'border-bottom': '20px solid transparent',  
                            'border-top': '10px solid transparent',     
                            'border-left': '0px solid transparent',  
                            'border-right': '0px solid transparent' 
                        }
                    )
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dcc.Graph(
                                    id='network-graph',
                                    figure=fig,
                                    style={'width': '900px', 'height': '900px'}  # Set specific pixel dimensions
                                ),
                            dbc.Col([
                                html.H3("Description"),
                                html.Div(id='clicked-data', children="Click on a node to learn more!"),
                                html.H3("Filters", className="mt-4"),
                                dbc.Row([
                                    dbc.Col([
                                       html.H6("Departments:"),
                                       dcc.Checklist(
                                           id='department-checklist-app',
                                           # Assuming 'create_slicer_options' and 'options_list' are defined elsewhere
                                           options=self.create_dept_slicer_options(options_list),
                                           value=options_list,
                                           inline=False,
                                           inputStyle={"margin-right": "5px"}
                                       )
                                    ],
                                    width = 3, style={"margin-bottom": "20px"}),
                                    dbc.Col([
                                       html.H6("Integration Type:"),
                                       dcc.Checklist(
                                           id='integration-type',
                                           # Assuming 'create_slicer_options' and 'options_list' are defined elsewhere
                                           options=self.create_connection_slicer_options(),
                                           value=["Proprietary", "Manual", "Purchased"],
                                           inline=False,
                                           inputStyle={"margin-right": "5px"}
                                       ),  
                                    #  spacer                          
                                       html.H6("Time Frame:", style={"margin-top": "30px"}),
                                       dcc.Checklist(
                                           id='status',
                                           # Assuming 'create_slicer_options' and 'options_list' are defined elsewhere
                                           options=[{'label': "Past", 'value': "Past"},
                                                  {'label': "Present", 'value': "Present"},
                                                  {'label': "Future", 'value': "Future"}],
                                           value=["Present"],
                                           inline=False,
                                           inputStyle={"margin-right": "5px"}
                                       )
                                    ],
                                    width = 3, style={"margin-bottom": "20px"}),
                                ]),
                                dbc.Row(
                                    dbc.Col(
                                    html.Div(
                                        daq.BooleanSwitch(
                                            id='sensitivity-boolean',
                                            on=False,
                                            label="Sensitive-Only",
                                            labelPosition="bottom"
                                        ),
                                        className="d-flex"
                                    ),
                                    width={"size": 3, "offset": 0},  # Adjust the size to shrink the switch's column
                                    className="d-flex align-items-center",  # This will vertically center the switch
                                    ),
                                ),
                            ],
                            # stops column from squishing together
                            style={"minWidth": "550px", "whiteSpace": "normal", "margin-bottom": "20px"})
                        ])
                    ])
                ])
            ],
            fluid=True
        )

        # drilldown text
        @app.callback(
            Output('clicked-data', 'children'),
            Input('network-graph', 'clickData')
        )
        def display_click_data(clickData):
            if clickData is None:
                return html.Div("Click on a hoverable to learn more!", style={'margin-top': '20px'})

            # Extract data from the clicked point
            point_data = clickData['points'][0]
            customdata=point_data.get('customdata')
            node_label = point_data['hovertext']  # Assuming hovertext holds the node label

            # Fetch more info based on the clicked node (replace this with your logic)
            department_driver = self.name_department_dict.get(customdata).lower().capitalize()
            connection_type = self.integration_name_connecttype_dict.get(customdata)
            data_type = self.separate_list_keep_str(self.name_datatype_dict.get(customdata))
            description = f"Description: {self.name_description_dict.get(customdata)}"
            description_text_wrapped = self.dash_wrap_paragraph(description)
            # print(customdata, department_driver, connection_type)
            if connection_type != None:
                drilldown_text = [
                html.H5(f"{node_label}"),
                html.P(f"Department Driver: {department_driver}"),
                html.P(f"Connection Type: {connection_type}")]
            else:
                drilldown_text = [
                html.H5(f"{node_label}"),
                html.P(f"Department Driver: {department_driver}")]

            for line in description_text_wrapped:
                # adjusting the style makes the line spacing left, so it looks connected
                drilldown_text.append(html.P(line, style={'margin': '0.25em 0'}))
            
            drilldown_text.extend([
                html.P(f"Data Type: {data_type}", style={'margin-top': '17px', 'margin-bottom':'0px'}),
                html.A(
                    "Parameter Definitions",
                    href="https://publish.smartsheet.com/c7c9fdb5e54048d58f72bf1f5c2754bd",
                    target="_blank",
                    style={'font-size': '12px'}  # You can adjust the size as needed
                )
            ])

            # Format the information for display
            return html.Div(drilldown_text)
        
        # slicer updates
        @app.callback(
            # grabs slicer values
            Output('network-graph', 'figure'),
            [
                Input('sensitivity-boolean', 'on'),
                Input('department-checklist-app', 'value'),
                Input('status', 'value'),
                Input('integration-type', 'value'),
            ]
        )
        def update_graph(is_sensitive_only, selected_app_departments, selected_status, selected_integration_type):
            # print(selected_app_departments)
            connection_type_slicer_dict={
                "Proprietary":["Home-grown API"], 
                "Manual": ["Manual", "CSV Upload", "CLI"], 
                'Purchased':["IPaas", "Prebuilt Connector"]
            }
            timeframe_slicer_dict={
                "Past":[ "Past Integration"], 
                "Present": ["Stable", "Semi Stable", "Soon to be replaced"], 
                "Future":["Concept phase", "In development"]
            }
            filtered_inputs = self.graph_inputs
            # dept filter needs to go first
            filtered_inputs = self.filter_apps_bydept(selected_app_departments, filtered_inputs)
            filtered_inputs = self.generic_filter(selected_status, timeframe_slicer_dict, 'Status', filtered_inputs)
            filtered_inputs = self.generic_filter(selected_integration_type, connection_type_slicer_dict, 'Connection Type', filtered_inputs)
            # filter sensative
            if is_sensitive_only:   
                sensitivity_mask = filtered_inputs['Data Types'].apply(self.is_sensitive_data_type)
                filtered_inputs = filtered_inputs[sensitivity_mask]
                # print(filtered_inputs)
            fig= self.handle_visual(filtered_inputs)
            fig.update_layout(transition_duration=250)
            return fig
#endregion

# if __name__ == "__main__":
# initialize
print('initializing...')
config = {
    'airtable_token': os.environ.get('airtable_token'),
}
av = AirtableVisualizer(config)
# fetch data
print("fetching data...")
inputs = av.grab_data()
fig = av.handle_visual(inputs)
# run Dash App
print('running Dash...')
av.run_dash(app, fig)
