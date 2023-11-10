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
                time.sleep(2 ** retry_count)  # Exponential backoff
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
            time.sleep(1)
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
    def generate_dict_from_table(self, table, key_field, value_field):
        '''used to make dictionaries like name/biz function dict that can decide color/size of nodes/edges'''
        primary_table_id, list_of_secondary_table_ids = self.generate_pns_table_ids(table)
        records= self.completely_populate_one_table(primary_table_id,list_of_secondary_table_ids)        

        table_df = pd.DataFrame(records)
        return {key: value for key, value in zip(table_df[key_field], table_df[value_field])}    
    def generate_pipedict_from_table(self, table, key_field1, key_field2, value_field):
        '''used to make dictionaries like name/biz function dict that can decide color/size of nodes/edges'''
        primary_table_id, list_of_secondary_table_ids = self.generate_pns_table_ids(table)
        records= self.completely_populate_one_table(primary_table_id,list_of_secondary_table_ids)        

        table_df = pd.DataFrame(records)
        return_dict =  {f"{key}|{key2}": value for key, key2, value in zip(table_df[key_field1], table_df[key_field2], table_df[value_field])}
        return_dict.update({f"{key2}|{key}": value for key, key2, value in zip(table_df[key_field1], table_df[key_field2], table_df[value_field])})
        return return_dict
    def grab_data(self):
        '''grabs data and creates a set of objects that will help translate into components of the visual down below. 
        b/c bizfunc coordinates the color, we need those, and then conntectiontype will coordinate the lines so we need those, etc...
        returns self.df because that is assumed to be the graph inputs. This allows the graph to rerender with different inputs b/c the graphing equation requires inputs, instead of just assuming the input is self.df and there fore cannot be reconfigured (with slicers)'''
        self.graph_inputs = self.generate_main_df("Application Connections")
        self.app_bizfunc_dict = self.generate_dict_from_table('Applications', "Name", "Business Function")
        self.integration_tofrom_name_dict=self.generate_pipedict_from_table('Application Connections', 'Application Data From', "Application Data To", "Name")
        self.integration_name_bizfunc_dict=self.generate_dict_from_table('Application Connections', "Name", "Business Function")
        self.integration_tofrom_connecttype_dict=self.generate_pipedict_from_table('Application Connections', 'Application Data From', "Application Data To", "Connection Type")
        self.name_description_dict = self.generate_dict_from_table('Application Connections', "Name", "Further Comments")
        self.name_description_dict.update(self.generate_dict_from_table('Applications', "Name", "Function"))
        self.name_department_dict = self.generate_dict_from_table('Application Connections', "Name", "Department Driver")
        self.name_department_dict.update(self.generate_dict_from_table('Applications', "Name", "Department Driver"))
        self.integration_name_department_dict = self.generate_dict_from_table('Application Connections', "Name", "Department Driver")
        self.integration_name_connecttype_dict=self.generate_dict_from_table('Application Connections', "Name", "Connection Type")
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
    def adjust_connection_position(self, pos, connection, dx, dy):
        """
        Adjusts the positions of nodes and edges of a particular connection by 
        a specified x and/or y amount.

        Parameters:
            G (nx.Graph): The graph.
            pos (dict): The positions of the nodes.
            connection (tuple): The connection to adjust as a tuple of node names (node1, node2).
            dx (float): The x adjustment.
            dy (float): The y adjustment.
        """
        node1, node2 = connection
        pos[node1] = (pos[node1][0] + dx, pos[node1][1] + dy)
        pos[node2] = (pos[node2][0] + dx, pos[node2][1] + dy)
        return pos
    def remove_invisble_edges(self, G, new_connetions):
        '''Connections were fabricated to produce clean pos (positions) on the graph. Once the positions are generated, we want to delete the new connections, as they are not based on the input'''
        for connection in new_connetions:
            if G.has_edge(connection[0], connection[1]):
                G.remove_edge(connection[0], connection[1])
    def adjust_node_position(self, pos, node, dx, dy):
        # Update the position of the specified node
        pos[node] = (pos[node][0] + dx, pos[node][1] + dy)
        return pos
    def handle_adjustments(self, adjustments, pos):
        '''takes an adjustments object and makes all adjustments as a one-liner
        if the key is a tuple, moves both nodes and the edge between them, nothing else (not v useful)
        if the key is a string, moves that node and all connected edges'''
        # if no adjustments, returns existing position and continues
        adjusted_pos = pos

        for nodes, adj_values in adjustments.items():
            try:
                x_adj = adj_values['x']
                y_adj = adj_values['y']
                if isinstance(nodes, tuple):
                    adjusted_pos = self.adjust_connection_position(pos, nodes, x_adj, y_adj)
                else:
                    adjusted_pos = self.adjust_node_position(pos, nodes, x_adj, y_adj)
            except KeyError:
                # wont work if the adjustment was filtered out!
                pass
        return adjusted_pos
    def wrap_text(self, input):
        '''wraps text but replacing spaces with line breaks, making every new space put text on new line'''
        text=str(input)
        return text.replace(' ', '<br>')
    def return_colormap(self):
        '''explain'''
        # WHY IS THIS HARDCODED???
        unique_business_functions = ["Business Function Scripting", 
                                     "Enhanced Productivity", 
                                     "Safety & Compliance", 
                                    #  "CRM", 
                                     "File Management", 
                                    #  "Issue Tracking/Support System", 
                                     "Financial Management", 
                                     "User Friendly Database", 
                                     "Communication", 
                                    #  "Business Intelligence", 
                                    #  "Engineering", 
                                     "Employee Insurance", 
                                     "Human Resource Management", 
                                    #  "Recruiting",
                                    #  "Employee Investment",
                                    #  "Contract Management",
                                     "Security",
                                     "IT Ticket Management",
                                     "Learning/Training",
                                     "Asset Creation/Editing",
                                     "Data Management",
                                     "Marketing"]


        # Determine the number of colors needed
        n_colors = max(len(unique_business_functions), 20)

        colors_per_palette = n_colors // 3
        remaining_colors = n_colors % 3  # To handle the case where n_colors is not divisible by 3
        color_palette = (
            sns.color_palette("pastel", colors_per_palette) +
            sns.color_palette("Set2", colors_per_palette) +
            sns.color_palette("Set3", colors_per_palette + remaining_colors)  # Add the remaining colors to one of the palettes
        )

        color_map = {
            business_function: color_palette[i]
            for i, business_function in enumerate(unique_business_functions)
        }
        return color_map
    def get_color(self, biz_func):
        '''returns a color per biz func'''
        if biz_func is None:
            return 'rgba(200,200,200,1)'  # Default color for nodes with no matching data
        self.color_map = self.return_colormap()
        color = self.color_map[biz_func]
        return f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
    def get_line_style(self,edge):
        '''maps a particular line style per connection'''
        line_style_map = {
            'Prebuilt Connector': {'color': '#8de5a1', 'line': 'solid'},
            'CLI': {'color': '#8de5a1', 'line': 'solid'},
            'IPaas': {'color': '#888', 'line': 'solid'},
            'VSC Uploads': {'color': '#888', 'line': 'dash'},
            'Manual': {'color': '#ff9f9b', 'line': 'dash'},
            'Home-grown API': {'color': '#888', 'line': 'solid'},
            # 'CONNECT CLUSTERS': {'color': 'rgba(0, 0, 0, 0)', 'line': 'dash'}
        }
        connection_type = self.integration_tofrom_connecttype_dict.get(f"{edge[0]}|{edge[1]}")
        return line_style_map.get(connection_type, {'color': '#888', 'line': 'solid'})
    def handle_line_style(self, G):
        '''groups the line styles'''
        grouped_edges = defaultdict(list)
        for edge in G.edges():
            style = self.get_line_style(edge)
            grouped_edges[(style['color'], style['line'])].append(edge)
        return grouped_edges
    #endregion
    def load_data_into_visual(self, graph_inputs):
        '''first step to plotly, data should be df'''
        # should be subset of 1st or second column, not a column w specific name!
        self.df = graph_inputs.dropna(subset=['Application Data To'])
        initial_G = nx.from_pandas_edgelist(self.df, 'Application Data From', 'Application Data To')
        new_connections = self.handle_disconnected_clusters(initial_G)
        self.G = nx.from_pandas_edgelist(self.df, 'Application Data From', 'Application Data To')
        pos = nx.kamada_kawai_layout(self.G)
        self.remove_invisble_edges(self.G, new_connections)
        return self.G, pos
    def handle_edges(self, G, pos):
        '''this handles edges, and the nodes in the middle of the edge (mnode)'''

        grouped_edges = self.handle_line_style(G)
        traces = []

        for (color, line), edges in grouped_edges.items():
            edge_x, edge_y, mnode_x, mnode_y, mnode_txt, mnode_colors, connection_name = [], [], [], [], [], [], []

            for edge in edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                mnode_x.extend([(x0 + x1) / 2])
                mnode_y.extend([(y0 + y1) / 2])
                integration_name = self.integration_tofrom_name_dict.get(f"{edge[0]}|{edge[1]}")
                integration_business_function = self.integration_name_bizfunc_dict.get(integration_name)
                # print(integration_name, integration_business_function)
                mnode_colors.append(self.get_color(integration_business_function))
                mnode_txt.extend([f'{integration_name}'])
                connection_name.append(integration_name)

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color=color, dash=line),
                hoverinfo='text',
                hovertext="doesn't work",
                mode='lines'
            )
            traces.append(edge_trace)

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
                    opacity=0.5
                )
            )
            traces.append(mnode_trace)

        return traces
    def handle_nodes(self, G, pos):
        '''explain'''

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_colors = [self.get_color(self.app_bizfunc_dict.get(app)) for app in G.nodes()]

        node_adjacencies = []
        node_text = []
        app_name=[] #for custom metadata that can help w/onclicks
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            # Generate hover text for each node (app, Biz Func - # Connections)
            # node_text.append(f'{adjacencies[0]}, {self.app_bizfunc_dict.get(adjacencies[0])} - {str(len(adjacencies[1]))} connections')  # adjacencies[0] holds the name of the application
            node_text.append(f'{adjacencies[0]}, {self.app_bizfunc_dict.get(adjacencies[0])}')
            app_name.append(adjacencies[0])
        # adjusted_zoom_fathom = adjust_connection_positions(G, pos, ("Zoom", "Fathom"), 0.3, 0)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            # text=[adjacencies[0] for adjacencies in G.adjacency()],  # Add node labels
            customdata=app_name,
            textposition='middle center',  # Centers text in circles
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                color=node_colors,
                size=37,
                line_width=1)
            )

        node_text_wrapped = [self.wrap_text(node) for node in G.nodes()]

        self.annotations = [
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
        return node_trace
    def generate_viz(self, edge_traces, node_trace):
        '''explain'''
        fig = go.Figure(data=edge_traces + [node_trace], 
                        layout=go.Layout(
                            # title='<br>Network Graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                        hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=2),
                            annotations=self.annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=800,  # Example width
                            height=800  # Example height, equal to width to make it square
                        ))
        return fig
    def create_legend_old(self):
        '''explain'''
        legend_data = []
        for i, (label, color) in enumerate(self.color_map.items()):
            legend_data.append(
                go.Scatter(
                    x=[1],
                    y=[len(self.color_map) - i],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='rgb({}, {}, {})'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    ),
                    name=label,
                    hoverinfo='none'
                )
            )

        layout = go.Layout(
            # title="Legend",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                automargin=True
            ),
            margin=dict(l=2, r=2, t=2, b=10),
            # margin=dict(l=60, r=10, t=40, b=10),
            height=400,
            width=300,
        )

        return go.Figure(data=legend_data, layout=layout)   
    def create_legend(self):
        '''Create a legend figure without an accompanying graph.'''
        legend_data = [
            go.Scatter(
                x=[None],  # No actual data points
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
                ),
                name=label,
                hoverinfo='none'
            ) for i, (label, color) in enumerate(self.color_map.items())
        ]

        layout = {
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
            'height': 400,
            'width': 300
        }

        return go.Figure(data=legend_data, layout=layout)
    def handle_visual(self, graph_inputs):
        G, adjusted_pos = self.load_data_into_visual(graph_inputs)
        # adjustments = {
            # ("Zoom", "Fathom"): {"x": 0.4, "y": 0},
        # "Airtable": {"x": -.1, "y": -.05},
        # "BambooHR": {"x": -.05, "y": -.05},
        # "Fieldwire": {"x": .03, "y": -.03}
        # }
        # adjusted_pos = self.handle_adjustments(adjustments, pos)
        edge_traces = self.handle_edges(G, adjusted_pos)
        node_trace = self.handle_nodes(G, adjusted_pos)
        fig = self.generate_viz(edge_traces, node_trace)
        legend_fig = self.create_legend()
        # return fig, legend_fig, 
        return fig, legend_fig
    def show_locally(self, fig, legend_fig):
        '''shows graph locally'''
        fig.show()
        legend_fig.show()
#endregion
#region dash (configure dynamic frontend)
    #region dash helpers
    def create_slicer_options(self, options_list):
        '''explain'''
        options = [{'label': option, 'value': option} for option in options_list]
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
    #endregion
    def run_dash(self, app, fig, legend_fig):
        '''explain'''

        options_list = [value for value in list(set(self.integration_name_department_dict.values())) if str(type(value)) == "<class 'str'>"]

        app.layout = dbc.Container(
            [
                dbc.Row(
                    dbc.Col(
                        html.H2("IT Application Integration Map", className="text-center"),
                        width=12
                    ),
                    className="mb-4"  # Adjust bottom margin as needed
                ),
                dbc.Row(
                    [
                        dcc.Checklist(
                            id='department-checklist',
                            options=self.create_slicer_options(options_list),
                            value=options_list,  # Default value
                            inline=True,
                            inputStyle={"margin-right": "3px", "margin-left": "6px"}  # Adds space around the checkbox itself
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id='network-graph',
                                figure=fig,
                                style={'width': '800px', 'height': 'auto'}  # Set the width to 800px and let height adjust automatically
                            ),
                            style={'paddingRight': 0, 'paddingLeft': 0}  # Remove padding from the column
                        ),
                        dbc.Col(
                            [
                                html.H3("Description"),
                                html.Div(id='clicked-data'),  # This div will display additional info on click
                                html.H3("Legend", className="mt-4"),  # Add margin-top for spacing
                                dcc.Graph(
                                    id='legend',
                                    figure=legend_fig,
                                    config={'displayModeBar': False}
                                ),
                            ],
                        ),
                    ],
                    className="align-items-start"  # Align the tops of the columns
                ),
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
            description = f"Description: {self.name_description_dict.get(customdata)}"
            description_text_wrapped = self.dash_wrap_paragraph(description)
            # print(customdata, department_driver, connection_type)
            if connection_type != None:
                drilldown_text = [
                html.H5(f"Node: {node_label}"),
                html.P(f"Department Driver: {department_driver}"),
                html.P(f"Connection Type: {connection_type}")]
            else:
                drilldown_text = [
                html.H5(f"Node: {node_label}"),
                html.P(f"Department Driver: {department_driver}")]

            
            for line in description_text_wrapped:
                # adjusting the style makes the line spacing left, so it looks connected
                drilldown_text.append(html.P(line, style={'margin': '0.25em 0'}))


            # Format the information for display
            return html.Div(drilldown_text)
        
        # slicer updates
        @app.callback(
            Output('network-graph', 'figure'),
            Input('department-checklist', 'value')
        )
        def update_graph(selected_departments):
            # Filter your graph data based on the selected_departments

            filtered_inputs = self.graph_inputs[self.graph_inputs['Department Driver'].isin(selected_departments)]
            fig, legend_fig = self.handle_visual(filtered_inputs)
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
fig, legend_fig = av.handle_visual(inputs)
# run Dash App
print('running Dash...')
av.run_dash(app, fig, legend_fig)
