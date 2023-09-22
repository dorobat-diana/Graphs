import os

from Graph import *
class UI:
    def __init__(self):
        self.graph = graph()
    def read_number(self,message):
        number = input(message)
        #verify if number is a int
        while not number.lstrip('-').isdigit():
            print("The number must be a number!")
            number = input(message)
        #make number a int
        number=int(number)
        return number

    def empty_graph(self):
        self.graph = graph()
        print("Empty graph created.")
    def nm_graph(self):
        n = self.read_number("Number of vertices: ")
        m = self.read_number("Number of edges: ")
        try:
            if n>=0 and m>=0:
                self.graph = random_graph(n,m)
                print("Graph created.")
            else:
                print("only positive numbers!")
        except GraphError as e:
            print(e)

    def add_vertex(self):
        vertex = self.read_number("Vertex: ")
        try:
            if vertex>=0:
                self.graph.add_vertex(vertex)
                print("Vertex added.")
            else:
                print("only positive numbers!")
        except GraphError as e:
            print(e)

    def add_edge(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        c = self.read_number("Cost: ")
        try:
            self.graph.add_edge(x, y, c)
            print("Edge added.")
        except GraphError as e:
            print(e)

    def rem_vertex(self):
        x = self.read_number("Vertex: ")
        try:
            self.graph.remove_vertex(x)
            print("Vertex removed.")
        except GraphError as e:
            print(e)

    def rem_edge(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        try:
            self.graph.remove_edge(x, y)
            print("Edge removed.")
        except GraphError as e:
            print(e)

    def change_edge(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        c = self.read_number("New cost: ")
        try:
            self.graph.set_cost_on_position(x, y, c)
            print("Cost changed.")
        except GraphError as e:
            print(e)

    def in_degree(self):
        x = self.read_number("Vertex: ")
        try:
            print(self.graph.in_degree(x))
        except GraphError as e:
            print(e)
    def out_degree(self):
        x = self.read_number("Vertex: ")
        try:
            print(self.graph.out_degree(x))
        except GraphError as e:
            print(e)
    def cnt_vertices(self):
        print(self.graph.vertices)
    def cnt_edges(self):
        print(self.graph.edges)
    def is_edge(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        print(self.graph.is_edge(x, y))
    def print_vertex_list(self):
        for i in self.graph.vertices_iterator():
            print(i," ")
    def print_outbound_list(self):
        obs=0
        x = self.read_number("Vertex: ")
        for i in self.graph.outbound_edges(x):
            print(i," ")
            obs=1
        if obs==0:
            print(-1)

    def print_inbound_list(self):
        obs=0
        x = self.read_number("Vertex: ")
        for i in self.graph.inbound_edges(x):
            print(i," ")
            obs=1
        if obs==0:
            print(-1)
    def print_edges(self):
        anyone = False
        for i in self.graph.edges_iterator():
            print("Vertices {0}, {1} and cost {2}.".format(i[0], i[1], i[2]))
            anyone = True
        if not anyone:
            print("No edges in the graph.")
    def readfile(self):
        file = input("File: ")
        #verify if file exists
        while not os.path.isfile(file):
            print("The file doesn't exist!")
            file = input("File: ")
        try:
            self.graph = read_file(file)
        except GraphError as e:
            print(e)
        else:
            print("Graph created.")
    def write_file(self):
        file = input("File: ")
        #verify if file name is empty
        while file=="":
            print("The file name is empty!")
            file = input("File: ")
        try:
            write_file(file,self.graph)
        except GraphError as e:
            print(e)
        print("Graph written.")

    def print_dict(self):
        print("\nThe dictionary of vertices/nodes is: { ", end="")
        for node in self.graph.vertices_iterator():
            print(node, end=", ")
        print("}\n")

        print("The dictionary of inbounds is: { ", end="")
        obs=0
        for node in self.graph.vertices_iterator():
            obs=0
            print(node, end=": {")
            for neighbour in self.graph.inbound_edges(node):
                print(neighbour, end=", ")
                obs=1
            if obs==0:
                print(-1,end=", ")
            print(end="}")
            print("; ", end="")
        print("}\n")
        obs=0
        print("The dictionary outbound is: { ", end="")
        for node in self.graph.vertices_iterator():
            obs=0
            print(node, end=": {")
            for neighbour in self.graph.outbound_edges(node):
                print(neighbour, end=", ")
                obs=1
            if obs==0:
                print(-1,end=", ")
            print(end="}")
            print("; ", end="")
        print("}\n")

        print("The dictionary of edges with their costs is: { ", end="")
        for triple in self.graph.edges_iterator():
            print("({0},{1}): {2}, ".format(triple[0], triple[1], triple[2]), end="")
        print("}\n")

    def vertix_in_graph(self):
        x = self.read_number("Vertex: ")
        print(self.graph.is_vertex(x))

    def graph_copy(self):
        self.graph2 = self.graph.copy_graph()
        write_file("graphcopy.txt",self.graph2)
        print("Graph copied.")

    def lowest_length_path(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        try:
            path=self.graph.lowest_length_path(x,y)
        except GraphError as e:
            print(e)
        else:
            print("The lowest length path is: ")
            for i in path:
                print(i,"\n","|","\n","v")
            if len(path)==0:
                print("No path between the two vertices.")
            else:
                print("Lowest length path is of length",len(path)-1)

    def lowest_cost_path(self):
        x = self.read_number("First vertex: ")
        y = self.read_number("Second vertex: ")
        try:
            path,cost=self.graph.lowest_cost_path(x,y)
        except GraphError as e:
            print(e)
        else:
            print("The lowest cost path is: ")
            for i in path:
                print(i,"\n","|","\n","v")
            if len(path)==0:
                print("No path between the two vertices.")
            else:
                print("Lowest cost path is of length",cost)
    def higher_cost_path(self):
        try:
            self.graph.topological_sort()
        except GraphError as e:
            print(e)
        else:
            print("The graph is acyclic.\n")
            x = int(input("First vertex: "))
            y = int(input("Second vertex: "))
            try:
                path,cost=self.graph.highest_cost_path(x,y)
            except GraphError as e:
                print(e)
            else:
                print("The higher cost path is: ")
                for i in path:
                    print(i,"\n","|","\n","v")
                if len(path)==0:
                    print("No path between the two vertices.")
                else:
                    print("Higher cost path is of length",cost)

    def connected_components(self):
        print("The connected components are: ")
        for i in self.graph.find_connected_components_unsigned():
            print(i)

    def scc(self):
        print("The strongly connected components are: ")
        for i in self.graph.find_connected_components_signed():
            print(i)

    def biconnected_components(self):
        print("The biconnected components are: ")
        for i in self.graph.find_biconnected_components():
            print(i)
    def negative_cycle(self):
        try:
            vertex= self.read_number("Vertex: ")
            #verify if vertex is in graph
            while not self.graph.is_vertex(vertex):
                print("The vertex is not in the graph!")
                vertex = self.read_number("Vertex: ")
            print(self.graph.negative_cycle(vertex))
        except GraphError as e:
            print(e)
    def Floyd_Warshall(self):
        try:
            matrix=self.graph.FloydWarshall()
            for i in range(self.graph.vertices):
                for j in range(self.graph.vertices):
                    print(matrix[i][j],end=" ")
                print()
        except GraphError as e:
            print(e)

    def nr_paths(self):
        try:
            x = self.read_number("First vertex: ")
            y = self.read_number("Second vertex: ")
            #verify if vertex is in graph
            while not self.graph.is_vertex(x):
                print("The vertex is not in the graph!")
                x = self.read_number("First vertex: ")
            while not self.graph.is_vertex(y):
                print("The vertex is not in the graph!")
                y = self.read_number("Second vertex: ")
            print(self.graph.count_paths(x,y))
        except GraphError as e:
            print(e)
    def Kruskal(self):
        try:
            print(self.graph.Kruskals_algorithm())
        except GraphError as e:
            print(e)

    def Prime(self):
        try:
            print(self.graph.primMST())
        except GraphError as e:
            print(e)
    def start(self):
        commands = {"1": self.empty_graph,
                    "2": self.nm_graph,
                    "3": self.add_vertex,
                    "4": self.add_edge,
                    "5": self.rem_vertex,
                    "6": self.rem_edge,
                    "7": self.change_edge,
                    "8": self.in_degree,
                    "9": self.out_degree,
                    "10": self.cnt_vertices,
                    "11": self.cnt_edges,
                    "12": self.is_edge,
                    "13": self.print_vertex_list,
                    "14": self.print_outbound_list,
                    "15": self.print_inbound_list,
                    "16": self.print_edges,
                    "17": self.readfile,
                    "18": self.write_file,
                    "19": self.print_dict,
                    "20": self.vertix_in_graph,
                    "21": self.graph_copy,
                    "22": self.lowest_length_path,
                    "23": self.lowest_cost_path,
                    "24" : self.higher_cost_path,
                    "25": self.connected_components,
                    "26": self.scc,
                    "27": self.biconnected_components,
                    "28": self.negative_cycle,
                    "29": self.Floyd_Warshall,
                    "30": self.nr_paths,
                    "31": self.Kruskal,
                    "32": self.Prime}
        while True:
            print("1. Generate an empty graph")
            print("2. Generate a graph with n vertices and m random edges")
            print("3. Add a vertex")
            print("4. Add an edge")
            print("5. Remove a vertex")
            print("6. Remove an edge")
            print("7. Change the cost of an edge")
            print("8. Print the in degree of a vertex")
            print("9. Print the out degree of a vertex")
            print("10. Print the number of vertices")
            print("11. Print the number of edges")
            print("12. Check whether an edge belongs to the graph")
            print("13. Print the list of vertices")
            print("14. Print the list of outbound neighbours of a vertex")
            print("15. Print the list of inbound neighbours of a vertex")
            print("16. Print the list of edges")
            print("17. Reads the graph from a file")
            print("18. Writes the graph to a file")
            print("19. Print the dictionaries")
            print("20. Check if a vertex is in the graph")
            print("21. Copy the graph")
            print("22. Find the lowest length path between two vertices")
            print("23. Find the lowest cost path between two vertices")
            print("24. Find the higher cost path between two vertices")
            print("25. Find the connected components of the unsigned graph")
            print("26. Find strongly connected components of the signed graph")
            print("27. Find the biconnected components of the unsigned graph")
            print("28. Verify if a connected graph has any negative cycles")
            print("29. Get the matrix with all the shortest paths")
            print("30. Find the number of paths between two vertices")
            print("31. Construct a spanning tree using Kruskal's algorithm")
            print("32. Construct a spanning tree using Prim's algorithm"  )
            print("0. Exit")
            index = input("> ")
            if index in commands:
                commands[index]()
            elif index == "0":
                break
            else:
                print("Invalid choice.")


UI().start()
