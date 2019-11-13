from node2vec import Node2Vec
import networkx as nx
def main():
    G=nx.read_edgelist('./data/Wiki_edgelist.txt',
                        create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])#read graph

    model = Node2Vec(G, walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 1)#init model
    model.train(window_size = 5, iter = 3)# train model
    embeddings = model.get_embeddings()
    print(embeddings)

if __name__ == '__main__':
    main()