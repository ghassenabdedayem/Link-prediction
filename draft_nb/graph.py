import networkx as nx

def read_train_val_graph(path='../input_data/edgelist.txt', val_ratio=0.1):
    G = nx.read_edgelist('../input_data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edges = list(G.edges())

    print('Number of nodes of total set:', n)
    print('Number of edges of total set:', m)

    node_to_idx = dict()
    for i, node in enumerate(nodes):
        node_to_idx[node] = i

    val_edges = list()
    G_train = G

    for edge in edges:
        if random() < val_ratio:
            val_edges.append(edge)

    # We remove the val edges from the graph G
    for edge in val_edges:
        G_train.remove_edge(edge[0], edge[1])

    n = G_train.number_of_nodes()
    m = G_train.number_of_edges()
    train_edges = list(G_train.edges())

    print('Number of nodes of training set:', n)
    print('Number of edges of training set:', m)

    y_val = [1]*len(val_edges)

    n_val_edges = len(val_edges)

    # Create random pairs of nodes (testing negative edges)
    for i in range(n_val_edges):
        n1 = nodes[randint(0, n-1)]
        n2 = nodes[randint(0, n-1)]
        (n1, n2) = (min(n1, n2), max(n1, n2))
        val_edges.append((n1, n2))

    # Remove from val_edges edges that exist in both train and val
    # for edge in list(set(val_edges) & set(train_edges)):
    #     val_edges.remove(edge)

    # n_val_edges = len(val_edges) # - len(y_val) #because we removed from val_edges edges that exist in both
    y_val.extend([0]*(n_val_edges))
    return G_train, train_edges, val_edges, nodes