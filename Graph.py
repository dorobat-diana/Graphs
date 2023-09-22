import heapq
import sys
from collections import deque
from copy import deepcopy
from os import close

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
    for line in file:
        tokens = line.split()
        if len(tokens) == 2:
            vertex1 = int(tokens[0])
            try:
                g.add_vertex(vertex1)
            except GraphError:
                pass
        else:
            vertex1 = int(tokens[0])
            vertex2 = int(tokens[1])
            edge_cost = int(tokens[2])
            try:
                g.add_vertex(vertex1)
            except GraphError:
                pass
            try:
                g.add_vertex(vertex2)
            except GraphError:
                pass
            try:
                g.add_edge(vertex1, vertex2, edge_cost)
            except GraphError:
                pass
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
    # verify if the number of edges is valid
    if edges > vertices * (vertices - 1):
        raise GraphError("too many edges")
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
        self._time = 0

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

    def find_connected_components_unsigned(self):
        """Finds the connected components of an unsigned graph, in O(m+n)."""
        visited = set()
        components = []
        # Perform depth-first search on each vertex
        for vertex in self.vertices_iterator():
            if vertex not in visited:
                component = []
                self.dfs_util_unsigned(vertex, visited, component)
                components.append(component)
        return components

    def dfs_util_unsigned(self, v, visited, component):
        """Utility function for find_connected_components_unsigned."""
        visited.add(v)
        component.append(v)
        # Recursively visit all neighbors of the current vertex
        for neighbor in self.outbound_edges(v):
            if neighbor not in visited:
                self.dfs_util_unsigned(neighbor, visited, component)

    def dfs(self, v, visited, stack):
        # Mark current vertex as visited
        visited.add(v)
        # Recursively visit all neighbors of the current vertex
        for neighbor in self.outbound_edges(v):
            if neighbor not in visited:
                self.dfs(neighbor, visited, stack)
        # Add current vertex to the result stack
        stack.append(v)

    def dfs_reversed(self, v, visited, stack):
        # Mark current vertex as visited
        visited.add(v)
        # Recursively visit all neighbors of the current vertex
        for neighbor in self.inbound_edges(v):
            if neighbor not in visited:
                self.dfs_reversed(neighbor, visited, stack)
        # Add current vertex to the result stack
        stack.append(v)

    def find_connected_components_signed(self):
        """Finds the connected components of a signed graph, in O(m+n), using Kosaraju's algorithm."""
        visited = set()
        stack = []

        # Perform depth-first search on each vertex
        for vertex in self.vertices_iterator():
            if vertex not in visited:
                self.dfs(vertex, visited, stack)
        # Create the transpose graph
        visited = set()
        components = []

        # Perform depth-first search on each vertex in the transpose graph
        while len(stack) > 0:
            vertex = stack.pop()
            if vertex not in visited:
                component = []
                self.dfs_reversed(vertex, visited, component)
                components.append(component)
        return components

    def find_biconnected_components(self):
        """Finds the biconnected components of an undirected graph, in O(m+n) using Tarjan's algorithm."""
        visited = set()
        stack = []
        components = []
        low = {}
        disc = {}
        parent = {}
        for vertex in self.vertices_iterator():
            parent[vertex] = None  # Initialize all vertices with no parent
        time = 0
        # Perform depth-first search on each vertex
        for vertex in self.vertices_iterator():
            if vertex not in visited:
                self.dfs_util_biconnected(vertex, visited, stack, low, disc, parent, components)
        return components

    def dfs_util_biconnected(self, v, visited, stack, low, disc, parent, components):
        """Utility function for find_biconnected_components."""
        visited.add(v)
        disc[v] = self._time
        low[v] = self._time
        self._time += 1
        children = 0
        # Recursively visit all neighbors of the current vertex
        for neighbor in self.outbound_edges(v):
            if neighbor not in visited:
                children += 1
                parent[neighbor] = v
                stack.append([v, neighbor])
                self.dfs_util_biconnected(neighbor, visited, stack, low, disc, parent, components)
                low[v] = min(low[v], low[neighbor])
                if (parent[v] == None and children >= 1) or (parent[v] != None and low[neighbor] >= disc[v]):
                    component = []
                    while stack[-1] != [v, neighbor]:
                        component.append(stack.pop())
                    component.append(stack.pop())
                    components.append(component)

            elif neighbor != parent[v]:
                low[v] = min(low[v], disc[neighbor])
        return components

    def negative_cycle(self, vertex):
        """Finds the lowest cost path between two given vertices, in O(m*n) with Bellman-Ford's algorithm."""
        # Check if the vertices exist
        if not self.is_vertex(vertex):
            raise GraphError("Vertex does not exist")

        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * self.vertices
        dist[vertex] = 0

        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edges
        for _ in range(self.vertices - 1):
            # Update dist value and parent index of the adjacent vertices of
            # the picked vertex. Consider only those vertices which are still in
            # queue
            for u in self.vertices_iterator():
                for v in self.outbound_edges(u):
                    if dist[u] != float("Inf") and dist[u] + self.get_cost(u, v) < dist[v]:
                        dist[v] = dist[u] + self.get_cost(u, v)

        # Step 3: check for negative-weight cycles. The above step
        # guarantees shortest distances if graph doesn't contain
        # negative weight cycle. If we get a shorter path, then there
        # is a cycle.

        for u in self.vertices_iterator():
            for v in self.outbound_edges(u):
                if dist[u] != float("Inf") and dist[u] + self.get_cost(u, v) < dist[v]:
                    return ("Graph contains negative weight cycle")
        return ("Graph does not contain negative weight cycle")

    def FloydWarshall(self):
        """Finds the lowest cost path between two given vertices, in O(m*n) with Floyd-Warshall's algorithm."""
        # dist[][] will be the output matrix that will finally have the shortest
        # distances between every pair of vertices
        # Initialize the solution matrix same as input graph matrix. Or
        # we can say the initial values of shortest distances are based
        # on shortest paths considering no intermediate vertex.
        dist = [[float("Inf")] * self.vertices for _ in range(self.vertices)]
        for i in range(self.vertices):
            for j in range(self.vertices):
                if i == j:
                    dist[i][j] = 0
                try:
                    dist[i][j] = self.get_cost(i, j)
                except GraphError:
                    pass

        # Add all vertices one by one to the set of intermediate vertices.
        # ---> Before start of an iteration, we have shortest distances
        # between all pairs of vertices such that the shortest distances
        # consider only the vertices in set {0, 1, 2, .. k-1} as intermediate vertices.
        # ----> After the end of an iteration, vertex no. k is added to the set of intermediate
        # vertices and the set becomes {0, 1, 2, .. k}
        for k in range(self.vertices):
            # Pick all vertices as source one by one
            for i in range(self.vertices):
                # Pick all vertices as destination for the
                # above picked source
                for j in range(self.vertices):
                    # If vertex k is on the shortest path from
                    # i to j, then update the value of dist[i][j]
                    if dist[i][k] != float("Inf") and dist[k][j] != float("Inf") and dist[i][k] + dist[k][j] < dist[i][
                        j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

    def count_paths(self, source, destination):
        """
        Counts the number of paths between two given vertices, in O(m+n) with BFS, using the algorithm based on predecessor counters.
        """
        num_vertices = self.vertices
        count = [0] * num_vertices
        count[source] = 1  # There is one path from source to itself

        queue = deque([source])

        while queue:
            vertex = queue.popleft()

            for neighbor in self.outbound_edges(vertex):
                if count[neighbor] == 0:
                    queue.append(neighbor)
                    count[neighbor] = count[vertex]
                else:
                    count[neighbor] += count[vertex]

        return count[destination]

    def find(self, parent, i):
        """A utility function to find set of an element i (uses path compression technique)."""
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def Kruskals_algorithm(self):
        """Finds the minimum spanning tree of a graph, in O(m*log(n)) with Kruskal's algorithm."""
        # Create a list of all edges in the graph, sorted by increasing order of cost
        edges = []
        for e in self.edges_iterator():
            edges.append(e)
        edges.sort(key=lambda e: self.get_cost(e[0], e[1]))
        result = []
        i=0
        e=0
        parent = []
        rank = []
        # Create V subsets with single elements
        for node in range(self.vertices):
            parent.append(node)
            rank.append(0)
        # Number of edges to be taken is equal to V-1
        while e < self.vertices - 1:
            # Pick the smallest edge and increment the index for next iteration
            u, v ,w= edges[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            # If including this edge does't cause cycle, include it in result
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        return result

    def union(self, parent, rank, x, y):

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[y] = x
            rank[x] += 1

        # A utility function to find the vertex with
        # minimum distance value, from the set of vertices
        # not yet included in shortest path tree

    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in self.vertices_iterator():
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

        # Function to construct and print MST for a graph
        # represented using adjacency matrix representation

    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.vertices
        parent = [None] * self.vertices  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.vertices

        parent[0] = -1  # First node is always the root of

        for cout in range(self.vertices):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.vertices):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.is_edge(u,v) and self.get_cost(u,v) > 0 and mstSet[v] == False \
                        and key[v] > self.get_cost(u,v):
                    key[v] = self.get_cost(u,v)
                    parent[v] = u

        return parent