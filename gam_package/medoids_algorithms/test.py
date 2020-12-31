import heapq
import math
from typing import List
from collections import defaultdict, deque
from enum import Enum
import collections

class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = map(len, (heights, heights[0]))
        efforts = [[math.inf] * n for _ in range(m)]
        efforts[0][0] = 0
        heap = [(0, 0, 0)]
        while heap:
            effort, x, y = heapq.heappop(heap)
            if (x, y) == (m - 1, n - 1):
                return effort
            for r, c in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if m > r >= 0 <= c < n:
                    next_effort = max(effort, abs(heights[r][c] - heights[x][y]))
                    if efforts[r][c] > next_effort:
                        efforts[r][c] = next_effort
                        heapq.heappush(heap, (next_effort, r, c))

if __name__ == '__main__':


    solution = Solution()
    heights = [[1,2,2],[3,8,2],[5,3,5]]
    print(solution.minimumEffortPath(heights))