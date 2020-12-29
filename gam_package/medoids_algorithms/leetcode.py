import math
from typing import List
from collections import defaultdict, deque
from enum import Enum



class Solution(object):
    def findRedundantConnection(self, edges):
        graph = defaultdict(set)

        def dfs(start, target):
            if start not in visited:
                visited.add(start)
                if start == target: return True
                arr = [dfs(neighbor, target) for neighbor in graph[start]]
                return any(arr)

        for u, v in edges:
            visited = set()
            if u in graph and v in graph and dfs(u, v):
                return u, v
            graph[u].add(v)
            graph[v].add(u)

if __name__ == '__main__':


    solution = Solution()
    edges = [[1,2], [2,3], [3,4], [1,4], [1,5]]
    print(solution.findRedundantConnection(edges))