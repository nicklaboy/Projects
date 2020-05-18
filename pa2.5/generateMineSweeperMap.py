from definitionsForAgent import MineSweeper
from random import randint


class GenerateMineSweeperMap:
    def __init__(self, dimensions, numberOfMines, startingCoordinate, map):
        self.dimensions = dimensions
        self.startingCoordinate = startingCoordinate
        self.numberOfMines = numberOfMines
        self.mines = {}
        if map == -1:
            self.hidden_map = [[0 for row in range(dimensions)] for column in range(dimensions)]
        else:
            self.hidden_map = map
            for x in range(self.dimensions):
                for y in range(self.dimensions):
                    self.hidden_map[x][y] = map[x][y]
                    if self.hidden_map[x][y] == MineSweeper.MINE:
                        self.mines[(x, y)] = True
                    else:
                        self.mines[(x, y)] = False
        self.agent_map = [[MineSweeper.CLEAR for row in range(dimensions)] for column in range(dimensions)]
        self.minesResolvedByAgent = 0
        self.agent_died = 0

    def isMine_local_map(self, coordinate):
        (x, y) = coordinate
        return self.hidden_map[x][y] == MineSweeper.MINE

    def isCellInMap(self, coordinate):
        (x, y) = coordinate
        if (0 <= x < self.dimensions) and (0 <= y < self.dimensions):
            return True
        else:
            return False

    def adjacentMines(self, coordinate):
        (x, y) = coordinate  #
        neighboringMines = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                            (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        neighboringMines = list(filter(self.isCellInMap, neighboringMines))
        neighboringMines = list(filter(self.isMine_local_map, neighboringMines))
        return neighboringMines

    def markAdjacentMines(self):
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if self.hidden_map[x][y] == MineSweeper.MINE:
                    self.mines[(x, y)] = True
                else:
                    neighboringMines = self.adjacentMines((x, y))
                    self.hidden_map[x][y] = len(neighboringMines)
                    self.mines[(x, y)] = False

    def updateAgentMap(self, uncoveredCell):
        (x, y) = uncoveredCell
        if self.hidden_map[x][y] == MineSweeper.MINE:
            #print("Your agent unfortunately blew up...")
            self.agent_died = self.agent_died + 1
            return -1
        else:
            return self.hidden_map[x][y]

    def createMineSweeperMap(self):
        minesToCreate = self.numberOfMines
        while minesToCreate > 0:
            x = randint(0, self.dimensions)
            y = randint(0, self.dimensions)
            if (x, y) == self.startingCoordinate:
                continue
            elif x < self.dimensions and y < self.dimensions and self.hidden_map[x][y] != MineSweeper.MINE:
                self.hidden_map[x][y] = MineSweeper.MINE
                minesToCreate = minesToCreate - 1
        self.markAdjacentMines()

    def validate_agent_solution(self, agent_solution):
        failure = 0
        mines_found = 0
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if (not ((self.hidden_map[x][y] == MineSweeper.MINE and agent_solution[x][y] == MineSweeper.FLAG)
                         or self.hidden_map[x][y] == agent_solution[x][y])):
                    failure = failure + 1
                if self.hidden_map[x][y] == MineSweeper.MINE and agent_solution[x][y] == MineSweeper.FLAG:
                    mines_found = mines_found + 1

        if failure == 0:
            # print("Agent successfully found all", self.numberOfMines, "mines and died ", self.agent_died)
            self.minesResolvedByAgent = self.numberOfMines
        else:
            self.minesResolvedByAgent = mines_found
            # print("Agent failed but found", mines_found, "mines and died ", self.agent_died)

    def print_hidden_map(self):
        print(" ------------- HIDDEN MAP ------------- ")
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if not self.mines[(x, y)]:
                    print("| ", self.hidden_map[x][y], end="")
                else:
                    print("|  M", end="")
            print("|", end="")
            print()
        print(" ------------- END OF MAP ------------- ")
