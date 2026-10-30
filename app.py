import streamlit as st

# Force Matplotlib to use non-interactive backend to prevent blank/freezing screens
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Page Config (Must be the first Streamlit command)
st.set_page_config(page_title="Intelligent Logistics & DAA Optimizer", layout="wide")

st.title("📦 Intelligent Logistics Route & Inventory Cost Optimizer")
st.markdown("### University DAA Project Dashboard")
st.markdown("---")

# --- INITIALIZE SESSION STATE FOR DYNAMIC DATABASE ---
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame([
        {"Item ID": 10, "Warehouse ID": 1, "Item Name": "Medical Kits", "Stock Level": 150, "Unit Weight": 15, "Priority Value": 90, "Reorder Threshold": 50},
        {"Item ID": 11, "Warehouse ID": 1, "Item Name": "Solar Panels", "Stock Level": 80, "Unit Weight": 25, "Priority Value": 100, "Reorder Threshold": 30},
        {"Item ID": 12, "Warehouse ID": 2, "Item Name": "Dry Rations", "Stock Level": 40, "Unit Weight": 10, "Priority Value": 60, "Reorder Threshold": 60},
        {"Item ID": 13, "Warehouse ID": 3, "Item Name": "Water Filters", "Stock Level": 100, "Unit Weight": 8, "Priority Value": 40, "Reorder Threshold": 40},
        {"Item ID": 14, "Warehouse ID": 4, "Item Name": "Emergency Tents", "Stock Level": 60, "Unit Weight": 20, "Priority Value": 80, "Reorder Threshold": 25}
    ])

if 'graph_data' not in st.session_state:
    st.session_state.graph_data = {
        'Delhi Hub': {'Jaipur Depot': {'weight': 280, 'cost': 1200}, 'Agra Center': {'weight': 210, 'cost': 950}, 'Kanpur Hub': {'weight': 490, 'cost': 2100}},
        'Jaipur Depot': {'Delhi Hub': {'weight': 280, 'cost': 1200}, 'Kanpur Hub': {'weight': 550, 'cost': 2400}},
        'Agra Center': {'Delhi Hub': {'weight': 210, 'cost': 950}, 'Kanpur Hub': {'weight': 330, 'cost': 1500}},
        'Kanpur Hub': {'Jaipur Depot': {'weight': 550, 'cost': 2400}, 'Agra Center': {'weight': 330, 'cost': 1500}, 'Delhi Hub': {'weight': 490, 'cost': 2100}}
    }


# --- DIJKSTRA'S ALGORITHM ---
def dijkstra(graph, start, goal):
    import heapq
    if start not in graph or goal not in graph:
        return float("inf"), []
    queue = [(0, start, [start])]
    visited = set()
    
    while queue:
        (cost, current, path) = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return cost, path
        
        for neighbor, details in graph.get(current, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + details['weight'], neighbor, path + [neighbor]))
    return float("inf"), []

# --- 0/1 KNAPSACK ALGORITHM ---
def knapsack(items, capacity):
    n = len(items)
    if n == 0 or capacity <= 0:
        return 0, []
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
    
    for i in range(1, n + 1):
        name, w, v = items[i-1]
        for w_cap in range(1, capacity + 1):
            if w <= w_cap:
                dp[i][w_cap] = max(v + dp[i-1][w_cap - w], dp[i-1][w_cap])
            else:
                dp[i][w_cap] = dp[i-1][w_cap]
                
    w_cap = capacity
    selected_items = []
    for i in range(n, 0, -1):
        if dp[i][w_cap] != dp[i-1][w_cap]:
            name, w, v = items[i-1]
            selected_items.append(name)
            w_cap -= w
            
    return dp[n][capacity], selected_items

# --- UI SIDEBAR CONTROLS ---
st.sidebar.header("Navigation & Controls")
app_mode = st.sidebar.selectbox("Choose View", [
    "Route & Path Optimizer (Dijkstra)", 
    "Cargo Loader (Knapsack DP)", 
    "Database Management"
])

nodes = list(st.session_state.graph_data.keys())

if app_mode == "Route & Path Optimizer (Dijkstra)":
    st.subheader("🚚 Shortest Path Route Optimization (Dijkstra's Algorithm)")
    st.markdown("Computes the optimal delivery path and distance between regional distribution centers based on the live network graph.")
    
    if not nodes or len(nodes) < 2:
        st.warning("Please add at least 2 connected cities/hubs in the Database view to calculate routes.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source Warehouse", nodes, index=0)
        with col2:
            dest_index = min(1, len(nodes) - 1)
            destination = st.selectbox("Destination Hub", nodes, index=dest_index)
            
        if st.button("Calculate Optimal Route"):
            total_dist, optimal_path = dijkstra(st.session_state.graph_data, source, destination)
            
            m1, m2 = st.columns(2)
            m1.metric("Shortest Distance", f"{total_dist} KM" if total_dist != float("inf") else "Infinity")
            m2.metric("Route Stops", len(optimal_path) if optimal_path else 0)
            
            if total_dist == float("inf"):
                st.error(f"❌ **No Path Found:** There is no route connecting **{source}** to **{destination}**. Please check your network connections or add an edge between them in the Database view.")
            else:
                st.success(f"**Optimal Path Found:** {' ➡️ '.join(optimal_path)}")
                
                # Draw Network Graph
                fig, ax = plt.subplots(figsize=(6, 4))
                G = nx.Graph()
                for u, neighbors in st.session_state.graph_data.items():
                    for v, data in neighbors.items():
                        G.add_edge(u, v, weight=data['weight'])
                        
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', ax=ax)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
                
                path_edges = list(zip(optimal_path[:-1], optimal_path[1:]))
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
                st.pyplot(fig)
                plt.close(fig)

elif app_mode == "Cargo Loader (Knapsack DP)":
    st.subheader("📦 Vehicle Cargo Loading Optimization (0/1 Knapsack)")
    st.markdown("Maximizes cargo priority value given a strict vehicle weight capacity limit, reading directly from the interactive Inventory Database.")
    
    max_weight = st.slider("Vehicle Weight Capacity (KG)", 10, 100, 45)
    
    inventory_items = []
    for _, row in st.session_state.inventory_df.iterrows():
        inventory_items.append((row["Item Name"], int(row["Unit Weight"]), int(row["Priority Value"])))
    
    st.write("Current Inventory Pool (Editable in Database View):")
    st.dataframe(st.session_state.inventory_df, use_container_width=True)
    
    if st.button("Run Knapsack Optimization"):
        max_val, loaded = knapsack(inventory_items, max_weight)
        st.metric("Max Cargo Priority Value", max_val)
        if loaded:
            st.success(f"**Optimal Items to Load in Vehicle:** {', '.join(loaded)}")
        else:
            st.warning("No items fit within the selected weight capacity.")

else:
    st.subheader("🗄️ Database Management & Network Graph Configuration")
    st.markdown("Edit network routes and inventory items directly below. Changes instantly update the DAA algorithms and restock alerts.")
    
    st.markdown("#### 1. Network Routes & Distances Configuration (Feeds Dijkstra)")
    st.markdown("Define direct distance (weight in KM) and shipping cost between connected hubs.")
    
    # Flatten graph data into an editable dataframe for routes
    route_rows = []
    for u, neighbors in st.session_state.graph_data.items():
        for v, details in neighbors.items():
            route_rows.append({"From": u, "To": v, "Distance (KM)": details['weight'], "Cost ($)": details['cost']})
    
    routes_df = pd.DataFrame(route_rows)
    edited_routes = st.data_editor(routes_df, num_rows="dynamic", key="routes_editor", use_container_width=True)
    
    # Reconstruct graph from edited routes table and automatically register nodes
    new_graph = {}
    for _, row in edited_routes.iterrows():
        u, v = row.get("From"), row.get("To")
        dist = row.get("Distance (KM)")
        cost = row.get("Cost ($)")
        if u and v and pd.notna(dist) and pd.notna(cost):
            u, v = str(u).strip(), str(v).strip()
            if u not in new_graph:
                new_graph[u] = {}
            new_graph[u][v] = {'weight': float(dist), 'cost': float(cost)}
            if v not in new_graph:
                new_graph[v] = {}
    st.session_state.graph_data = new_graph

    st.markdown("---")
    
    st.markdown("#### 2. Inventory Table (Feeds Knapsack Algorithm & Restock Orders)")
    edited_inventory = st.data_editor(st.session_state.inventory_df, num_rows="dynamic", key="inventory_editor")
    st.session_state.inventory_df = edited_inventory
    
    # Check for low stock / missing order automatically
    low_stock_items = edited_inventory[edited_inventory["Stock Level"] <= edited_inventory["Reorder Threshold"]]
    if not low_stock_items.empty:
        st.warning(f"⚠️ **Stock Alert:** {len(low_stock_items)} item(s) are at or below their reorder threshold and require restock orders!")

