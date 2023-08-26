import heapq
from copy import deepcopy
from error import *
import random


def read_file(file_name):
    """
    read a graph from a file
    :param file_name:
    :return:
    """
    try:
        file = open(file_name, "rt")
    except IOError:
        raise GraphError("file does not exist")
    v, e = map(int, file.readline().split())
    g = graph(v, e)
    for i in range(e):
        vertex1, vertex2, edge_cost = map(int, file.readline().split())
        try:
            g.add_vertex(vertex1)
        except GraphError:
            pass
        try:
            g.add_vertex(vertex2)
        except GraphError:
            pass
        g.add_edge(vertex1, vertex2, edge_cost)
    file.close()
    return g


def write_file(file_name, g):
    """
    write a graph to a file
    :param file_name:
    :param g:
    :return:
    """
    graph1 = graph()
    graph1 = g.copy_graph()
    file = open(file_name, "wt")
    file.write("{} {}\n".format(graph1.vertices, graph1.edges))
    for i in graph1.vertices_iterator():
        obs = 0
        for j in graph1.outbound_edges(i):
            file.write("{} {} {}\n".format(i, j, graph1.get_cost(i, j)))
            obs = 1
        if obs == 0:
            file.write("{} {}\n".format(i, -1))

    file.close()


def random_graph(vertices, edges):
    """
    create a random graph
    :param vertices:
    :param edges:
    :return:
    """
    g = graph(vertices, edges)
    for j in range(vertices):
        g.add_vertex(j)
    for i in range(edges):
        vertex1 = random.randint(0, vertices - 1)
        vertex2 = random.randint(0, vertices - 1)
        edge_cost = random.randint(0, 100)
        while g.is_edge(vertex1, vertex2):
            vertex1 = random.randrange(vertices - 1)
            vertex2 = random.randrange(vertices - 1)
        g.add_edge(vertex1, vertex2, edge_cost)
    return g


class graph:
    def __init__(self, vertices=0, edges=0):
        self._vertices = vertices
        self._edges = edges
        self._list_vertices = set()
        self._list_costs = dict()
        self._invertices = dict()
        self._outvertices = dict()

    def vertices_iterator(self):
        """
        :return: an iterator for the vertices
        :return:
        """
        for i in self._list_vertices:
            yield i

    def outbound_edges(self, vertix):
        """
        get outbound edges of a vertix
        :param vertix:
        :return:
        """
        if not self.is_vertex(vertix):
            raise GraphError("vertix does not exist")
        for i in self._outvertices[vertix]:
            yield i

    def inbound_edges(self, vertix):
        """
        get inbound edges of a vertix
        :param vertix:
        :return:
        """
        if not self.is_vertex(vertix):
            raise GraphError("vertix does not exist")
        for i in self._invertices[vertix]:
            yield i

    def edges_iterator(self):
        """
        :return: an iterator for the edges
        :return:
        """
        for vertex, cost in self._list_costs.items():
            yield vertex[0], vertex[1], cost

    def is_vertex(self, vertex):
        """
        :return: true if the vertex exists
        :param vertex:
        :return:
        """
        return vertex in self._list_vertices

    def is_edge(self, vertix1, vertix2):
        """
        : returns true if there is an edge between vertix1 and vertix2
        :param vertix1:
        :param vertix2:
        :return:
        """
        return vertix1 in self._outvertices and vertix2 in self._outvertices[vertix1]

    @property
    def vertices(self):
        """
        :return: the number of vertices
        """
        return len(self._list_vertices)

    @property
    def edges(self):
        """
        :return: the number of edges
        :return:
        """
        return len(self._list_costs)

    def in_degree(self, vertix):
        """
        get in degree of a vertix
        :param vertix:
        :return:
        """
        if not self.is_vertex(vertix):
            raise GraphError("vertix does not exist")
        return len(self._invertices[vertix])

    def out_degree(self, vertix):
        """
        get out degree of a vertix
        :param vertix:
        :return:
        """
        if not self.is_vertex(vertix):
            raise GraphError("vertix does not exist")
        return len(self._outvertices[vertix])

    def get_cost(self, x, y):
        """
        :return: the cost of the edges
        :return:
        """
        if not self.is_vertex(x) or not self.is_vertex(y):
            raise GraphError("vertix does not exist")
        if not self.is_edge(x, y):
            raise GraphError("Edge doesn't exist")
        return self._list_costs[(x, y)]

    def set_cost_on_position(self, x, y, new_cost):
        """
        set the cost of the edges
        :param new_cost:
        :return:
        """
        if not self.is_vertex(x) or not self.is_vertex(y):
            raise GraphError("vertix does not exist")
        if y not in self._outvertices[x]:
            raise GraphError("Edge doesn't exist")
        self._list_costs[(x, y)] = new_cost

    def add_vertex(self, vertex):
        """
        add a vertex to the graph
        :param vertex:
        :return:
        """
        if self.is_vertex(vertex):
            raise GraphError("Cannot add a vertex which already exists.")
        self._list_vertices.add(vertex)
        self._invertices[vertex] = set()
        self._outvertices[vertex] = set()

    def add_edge(self, vertix1, vertix2, cost=0):
        """
        add an edge between vertix1 and vertix2
        :param vertix1:
        :param vertix2:
        :param cost:
        :return:
        """
        if not self.is_vertex(vertix1) or not self.is_vertex(vertix2):
            raise GraphError("vertix does not exist")
        if self.is_edge(vertix1, vertix2):
            raise GraphError("Edge already exists")
        if vertix1 not in self._invertices:
            self._invertices[vertix1] = set()
        if vertix2 not in self._invertices:
            self._invertices[vertix2] = set()
        if vertix1 not in self._outvertices:
            self._outvertices[vertix1] = set()
        if vertix2 not in self._outvertices:
            self._outvertices[vertix2] = set()
        self._invertices[vertix2].add(vertix1)
        self._outvertices[vertix1].add(vertix2)
        self._list_costs[(vertix1, vertix2)] = cost
        self._edges += 1

    def remove_edge(self, vertix1, vertix2):
        """
        remove an edge between vertix1 and vertix2
        :param vertix1:
        :param vertix2:
        :return:
        """
        if not self.is_vertex(vertix1) or not self.is_vertex(vertix2):
            raise GraphError("vertix does not exist")
        if self.is_edge(vertix1, vertix2):
            self._invertices[vertix2].remove(vertix1)
            self._outvertices[vertix1].remove(vertix2)
            del self._list_costs[(vertix1, vertix2)]
            self._edges -= 1
        else:
            raise GraphError("Edge does not exist")

    def remove_vertex(self, vertix):
        """
        remove a vertix
        :param vertix:
        :return:
        """
        if not self.is_vertex(vertix):
            raise GraphError("vertix does not exist")
        toremove = []
        for i in self._invertices[vertix]:
            toremove.append((i, vertix))
        for i in toremove:
            self.remove_edge(i[0], i[1])
        toremove = []
        for i in self._outvertices[vertix]:
            toremove.append((vertix, i))
        for i in toremove:
            self.remove_edge(i[0], i[1])
        del self._invertices[vertix]
        del self._outvertices[vertix]
        self._list_vertices.remove(vertix)
        self._vertices -= 1

    def copy_graph(self):
        """
        copy the graph
        :return:
        """
        return deepcopy(self)

    def lowest_length_path(self, start, end):
        """
        Returns the lowest length path between start and end using BFS.
        """
        # Check if the start and end vertices exist in the graph
        if not self.is_vertex(start) or not self.is_vertex(end):
            raise GraphError("Vertex does not exist")

        # If start and end vertices are the same, return a path with only that vertex
        if start == end:
            return [start]

        # Initialize the queue with the start vertex and the set of visited vertices with the start vertex
        queue = [start]
        visited = set()
        visited.add(start)

        # Initialize the dictionary of previous vertices (i.e., the parent of each vertex in the path)
        previous = {}

        # While there are vertices in the queue
        while len(queue) > 0:
            # Remove the first vertex in the queue and mark it as current
            current = queue.pop(0)

            # For each neighbor of the current vertex
            for i in self.outbound_edges(current):
                # If the neighbor has not been visited yet
                if i not in visited:
                    # Mark the current vertex as the parent of the neighbor
                    previous[i] = current

                    # If the neighbor is the end vertex, construct and return the path
                    if i == end:
                        path = []
                        while i != start:
                            path.append(i)
                            i = previous[i]
                        path.append(start)
                        path.reverse()
                        return path

                    # Otherwise, mark the neighbor as visited and add it to the queue
                    visited.add(i)
                    queue.append(i)

        # If there is no path between the start and end vertices, return an empty list
        return []

    def lowest_cost_path(self, vertex1, vertex2):
        """
        Find the lowest cost walk between two vertices using a backwards Dijkstra algorithm.
        :param vertex1: The starting vertex.
        :param vertex2: The ending vertex.
        :return: A tuple (path, cost) where path is a list of vertices representing the lowest cost path from vertex1 to vertex2
        and cost is the cost of that path.
        """
        # Check that vertex1 and vertex2 are both vertices in the graph
        if not self.is_vertex(vertex1) or not self.is_vertex(vertex2):
            raise GraphError("Vertex does not exist")

        # If the starting vertex is the same as the ending vertex, return a path of length 1 and a cost of 0
        if vertex1 == vertex2:
            return [vertex1], 0

        # Create dictionaries to store the previous vertex and the distance to each vertex
        prev = {}
        dist = {}

        # Initialize the priority queue with the ending vertex and a cost of 0
        q = []
        heapq.heappush(q, [0, vertex2])
        dist[vertex2] = 0

        # Continue until all vertices have been explored or the lowest cost path from vertex2 to vertex1 has been found
        while len(q) > 0:
            # Pop the vertex with the lowest distance so far
            vertex = heapq.heappop(q)[1]

            # If we've reached vertex1, we can stop and return the lowest cost path from vertex2 to vertex1
            if vertex == vertex1:
                # Create the path by backtracking from vertex1 to vertex2 using the prev dictionary
                path = []
                while vertex != vertex2:
                    path.append(vertex)
                    vertex = prev[vertex]
                path.append(vertex2)
                path.reverse()
                # Return the path and its cost
                return path, dist[vertex1]

            # Otherwise, explore its neighbors and update their distances if necessary
            for neighbor in self.outbound_edges(vertex):
                cost = self.get_cost(vertex, neighbor)
                if neighbor not in dist or dist[neighbor] > dist[vertex] + cost:
                    dist[neighbor] = dist[vertex] + cost
                    prev[neighbor] = vertex
                    heapq.heappush(q, [dist[neighbor], neighbor])

        # If we've explored all vertices and haven't found a path from vertex2 to vertex1, return an empty path and a cost of 0
        return [], 0


    def dfs_util(self, v, visited, stack, rec_stack):
        # Mark current vertex as visited and add it to recursion stack
        visited.add(v)
        rec_stack.add(v)
        # Recursively visit all neighbors of the current vertex
        for neighbor in self.outbound_edges(v):
            if neighbor not in visited:
                if self.dfs_util(neighbor, visited, stack, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        # Remove current vertex from recursion stack and add it to the result stack
        rec_stack.remove(v)
        stack.append(v)
        return False

    def topological_sort(self):
        visited = set()
        stack = []
        rec_stack = set()
        # Perform depth-first search on each vertex
        for vertex in self.vertices_iterator():
            if vertex not in visited:
                if self.dfs_util(vertex, visited, stack, rec_stack):
                    raise GraphError("Graph contains cycles and does not have a topological ordering")
        # Return the reversed stack as the topological order
        return stack[::-1]


    def highest_cost_path(self, vertex1, vertex2):
        """Verify if the corresponding graph is a DAG and performs a topological sorting of the activities using the algorithm based on depth-first traversal (Tarjan's algorithm).
        If it is a DAG, finds a highest cost path between two given vertices, in O(m+n)."""
        # Check if the vertices exist
        if not self.is_vertex(vertex1) or not self.is_vertex(vertex2):
            raise GraphError("Vertex does not exist")
        # If the source and destination vertices are the same, return the vertex and cost 0
        if vertex1 == vertex2:
            return [vertex1], 0
        prev = {}
        dist = {}
        q = []
        path = []
        heapq.heappush(q, [0, vertex1])
        dist[vertex1] = 0
        dist[vertex2] = 0
        while len(q) > 0:
            vertex = heapq.heappop(q)[1]

            if vertex == vertex2:
                path = []
                # Reconstruct the path from destination to source
                while vertex != vertex1:
                    path.append(vertex)
                    vertex = prev[vertex]
                path.append(vertex1)
                path.reverse()
            # Explore the neighbors of the current vertex
            for neighbor in self.outbound_edges(vertex):
                cost = self.get_cost(vertex, neighbor)
                if neighbor not in dist or dist[neighbor] < dist[vertex] + cost:
                    dist[neighbor] = dist[vertex] + cost
                    prev[neighbor] = vertex
                    heapq.heappush(q, [dist[neighbor], neighbor])
        # Return the highest cost path and the corresponding cost
        return path, dist[vertex2]
