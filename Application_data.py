import requests
from requests.structures import CaseInsensitiveDict 
import json
import time
import pandas as pd
from globals import airtable_token
import networkx as nx
import plotly.graph_objects as go
import colorsys
import seaborn as sns


class AirtableVisualizer():
    '''Explain Class'''
    def __init__(self, config):
        self.airtable_token = config.get("airtable_token")
#region airtable
    def pull_table_api(self, table_id, data = None, offset=None):
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

        response = requests.get(f"https://api.airtable.com/v0/appY3Ht2ufuvQ0f18/{table_id}", params=params, headers=headers)
        if response.status_code != 200:
            raise Exception(f'Failed to retrieve data: {response.content}')
        resp_dict = json.loads(response.content.decode('utf-8'))
        data.extend(resp_dict.get('records', []))
        if resp_dict.get('offset'):
            return self.pull_table_api(table_id, data, resp_dict.get('offset'))
        else:
            return data 
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
        '''explain'''
        self.df = self.generate_main_df("Application Connections")
        self.app_bizfunc_dict = self.generate_dict_from_table('Applications', "Name", "Business Function")
        self.integration_tofrom_bizfunc_dict=self.generate_pipedict_from_table('Application Connections', 'Application Data From', "Application Data To", "Business Function")
        self.integration_name_bizfunc_dict=self.generate_dict_from_table('Application Connections', "Name", "Business Function")
        self.integration_connecttype_dict=self.generate_dict_from_table('Application Connections', "Name", "Connection Type")
#endregion
#region plotly
    #region helpers
    def adjust_connection_position(self, G, pos, connection, dx, dy):
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
    def adjust_node_position(self, G, pos, node, dx, dy):
        # Update the position of the specified node
        pos[node] = (pos[node][0] + dx, pos[node][1] + dy)
        return pos
    def wrap_text(self, input):
        '''wraps text but replacing spaces with line breaks, making every new space put text on new line'''
        text=str(input)
        return text.replace(' ', '<br>')
    def return_colormap(self):
        '''explain'''
        unique_business_functions = ["Business Function Scripting", 
                                     "Enhanced Productivity", 
                                     "Safety & Compliance", 
                                     "CRM", 
                                     "File Management", 
                                     "Issue Tracking/Support System", 
                                     "Financial Management Systems", 
                                     "User Friendly Database", 
                                     "Communication", 
                                     "Business Intelligence", 
                                     "Engineering", 
                                     "Employee Insurance", 
                                     "Human Resource Management", 
                                     "Recruiting",
                                    #  "Employee Investment",
                                     "Contract Management",
                                     "Security",
                                     "Learning/Training",
                                     "Asset Creation/Editing",
                                     "Server/Data Management",
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
        color = self.return_colormap()[biz_func]
        return f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
    #endregion
    def load_data_into_visual(self, data):
        '''first step to plotly, data should be df'''
        self.df = data.dropna(subset=['Application Data To'])
        G = nx.from_pandas_edgelist(self.df, 'Application Data From', 'Application Data To')
        pos = nx.kamada_kawai_layout(G)
        return G, pos
    def handle_edges(self, G, pos):
        '''this handles edges, and the nodes in the minddle of the edge (mnode)'''
        #EDGES
        # Generating edge_trace based on the positions of nodes:
        adjusted_pos = self.adjust_connection_position(G, pos, ("Zoom", "Fathom"), 0.3, 0)
        adjusted_pos = self.adjust_node_position(G, pos, "Airtable", -.1, -.05)
        adjusted_pos = self.adjust_node_position(G, pos, "BambooHR", -.05, -.05)
        adjusted_pos = self.adjust_node_position(G, pos, "Fieldwire", .03, -.03)
        edge_x, edge_y = [], []
        mnode_x, mnode_y, mnode_txt, mnode_colors = [], [], [], []
        for edge in G.edges():
            x0, y0 = adjusted_pos[edge[0]]
            x1, y1 = adjusted_pos[edge[1]]
            edge_x.extend([x0, x1, None])  # None creates a segment, separating edges
            edge_y.extend([y0, y1, None])
            mnode_x.extend([(x0 + x1)/2]) # assuming values positive/get midpoint
            mnode_y.extend([(y0 + y1)/2]) # assumes positive vals/get midpoint
            integration_name = self.integration_tofrom_bizfunc_dict.get(f"{edge[0]}|{edge[1]}")
            integration_business_function = self.integration_name_bizfunc_dict.get(integration_name)
            mnode_colors.append(self.get_color(integration_business_function))
            mnode_txt.extend([f'{integration_name}']) # hovertext


        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            hovertext="doesn't work",
            mode='lines'
        )



        # idea to replace mid nodes with many midnodes that are invisible: https://stackoverflow.com/questions/74607000/python-networkx-plotly-how-to-display-edges-mouse-over-text
        mnode_trace = go.Scatter(
            x=mnode_x, 
            y=mnode_y, 
            mode="markers", 
            showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=mnode_txt, 
            marker=dict(
                color=mnode_colors,  # Using the generated color list
                opacity=0.5
            )
        )

        return edge_trace, mnode_trace
    def handle_nodes(self, G, pos):
        '''explain'''

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_colors = [self.get_color(self.app_bizfunc_dict.get(app)) for app in G.nodes()]

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            # Generate hover text for each node (app, Biz Func - # Connections)
            node_text.append(f'{adjacencies[0]}, {self.app_bizfunc_dict.get(adjacencies[0])} - {str(len(adjacencies[1]))} connections')  # adjacencies[0] holds the name of the application

        # adjusted_zoom_fathom = adjust_connection_positions(G, pos, ("Zoom", "Fathom"), 0.3, 0)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            # text=[adjacencies[0] for adjacencies in G.adjacency()],  # Add node labels
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
    def generate_viz(self, edge_trace, mnode_trace, node_trace):
        '''explain'''
        fig = go.Figure(data=[edge_trace, mnode_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network Graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=self.annotations,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            width=800,  # Example width
                            height=800  # Example height, equal to width to make it square
                        ))
        return fig
    def handle_visual(self):
        G, pos = self.load_data_into_visual(self.df)
        edge_trace, mnode_trace = self.handle_edges(G, pos)
        node_trace = self.handle_nodes(G, pos)
        fig = self.generate_viz(edge_trace, mnode_trace, node_trace)
        fig.show()


#endregion

if __name__ == "__main__":
    config = {
        'airtable_token':airtable_token,
    }
    av = AirtableVisualizer(config)
    av.grab_data()
    av.handle_visual()



# converting major variables ipynb to py
# df=av.df
# name_function_dict=av.app_bizfunc_dict
# connection_dict=av.integration_tofrom_bizfunc_dict
# integration_bizfunc_dict=av.integration_name_bizfunc_dict