from random import randint, random
from generateMineSweeperMap import GenerateMineSweeperMap
from definitionsForAgent import MineSweeper

class BasicAgent(GenerateMineSweeperMap):
    def __init__(self, dimensions, mines,startingCoordinate, map):
        super().__init__(dimensions, mines, startingCoordinate, map)
        if(map == None):
            self.createMineSweeperMap()

        self.startingCoordinate = startingCoordinate
        #self.print_hidden_map()
        self.flags = {}
        self.cells = {}
        self.isVisited = {}
        #print(self.startingCoordinate)
        self.startingCoordinate = startingCoordinate
        self.totalSafeCells = (dimensions*dimensions) - self.numberOfMines
        (x, y) = startingCoordinate
        self.agent_map[x][y] = self.updateAgentMap(startingCoordinate)
        for x_o in range(dimensions):
            for y_o in range(dimensions):
                self.isVisited[startingCoordinate] = True
                self.isVisited[(x_o, y_o)] = False
                self.cells[(x_o, y_o)] = self.agent_map[x][y]
        #self.output_agent_map()
        #print(self.isVisited)
        self.solve(dimensions)

    def solve(self, dimensions):

        resolvedMines = self.numberOfMines
        stack = [self.startingCoordinate]
        mineCells = []
        safeCells = [self.startingCoordinate]
        count = 0
        agentDead = 0
        allCells = []
        for x in range(dimensions):
            for y in range(dimensions):
                allCells.append((x,y))

        while(resolvedMines > 0 and len(stack) > 0):
            coordinate = stack.pop()
            (x,y) = coordinate
            cellsToUncover = []
            neighbors = self.adjacent_cells_agent_map(coordinate)
            clue = self.agent_map[x][y]
            #print("Coordinate: ", coordinate)
            if(self.agent_map[x][y] == 0):
                cellsToUncover = neighbors #if 0, reveal all neighbors
            else:
                known_cells = list(filter(self.isCellKnown, neighbors))
                if(len(known_cells) == 0):
                    cellsToUncover.append(self.random_select(neighbors,[],-1))
                else:
                    unknown_cells = list(filter(self.isCellUnknown, neighbors))
                    cells_flagged = list(filter(self.isCellFlagged, neighbors))
                    if ((clue - len(cells_flagged)) == len(unknown_cells)):
                        for cell in unknown_cells:
                            (x_f,y_f) = cell
                            self.agent_map[x_f][y_f] = MineSweeper.FLAG
                            if(cell not in mineCells):
                                mineCells.append(cell)
                    if((8 - clue) - len(known_cells) == len(unknown_cells)):
                        cellsToUncover = unknown_cells
            #print(cellsToUncover)
            #self.output_agent_map()
            for cells_Uncover in cellsToUncover:
                if(self.isVisited[cells_Uncover] == False):
                    self.isVisited[cells_Uncover] = True
                    safeOrNot = self.updateAgentMap(cells_Uncover)
                    if(safeOrNot == -1):
                        agentDead = agentDead+1
                        if(cells_Uncover not in mineCells):
                            mineCells.append(cells_Uncover)
                        (x_f, y_f) = cells_Uncover
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    else:
                        (x_o, y_o) = cells_Uncover
                        self.agent_map[x_o][y_o] = safeOrNot
                        stack.append(cells_Uncover)
                        safeCells.append(cells_Uncover)
            if (len(stack) == 0):
                neighbors_new = neighbors
                while (len(neighbors_new) > 0):
                    restart = self.random_select(neighbors_new, [], -1)
                    (x_o, y_o) = restart
                    if(self.agent_map[x_o][y_o] == MineSweeper.FLAG):
                        neighbors_new.remove(restart)
                        continue
                    success = self.updateAgentMap(restart)
                    self.isVisited[restart] = True
                    if (success > -1):
                        self.agent_map[x_o][y_o] = success
                        stack.append(restart)
                        safeCells.append(restart)
                        break
                    else:
                        agentDead = agentDead + 1
                        if(restart not in mineCells):
                            mineCells.append(restart)
                            self.agent_map[x_o][y_o] = MineSweeper.FLAG
                        neighbors_new.remove(restart)

            if(self.numberOfMines == len(mineCells)):
                cellsLeft = list(filter(self.isCellUnknown, allCells))
                for cell in cellsLeft:
                    (x_f,y_f) = cell
                    success = self.updateAgentMap(cell)
                    if(success > -1):
                        self.agent_map[x_f][y_f] = success
                    else:
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                break
            count += 1
            #print("M: ", len(mineCells))
            resolvedMines = self.numberOfMines - len(mineCells)
        #print("Agent died ", agentDead ,"times")
        #print("Number of Mines found: ", len(mineCells))
        #self.output_agent_map()
        self.validate_agent_solution(self.agent_map)
        #if(wasMapSolvedSuccessfully):
         #   print("Agent solved map successfully!")
        #else:
            #print("Agent was unable to solve map")

    def random_select(self, neighbors, covered, p):
        index = randint(0, len(neighbors) - 1)
        if (len(covered) > 0):
            return covered[index]
        else:
            return neighbors[index]
    def isCellUnknown(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] == MineSweeper.CLEAR
    def isCellKnown(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] != MineSweeper.CLEAR and self.agent_map[x][y] != MineSweeper.FLAG

    def isCellFlagged(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] == MineSweeper.FLAG

    def adjacent_cells_agent_map(self, coordinate):
        (x, y) = coordinate #
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                     (x + 1, y + 1),(x + 1, y - 1),(x - 1, y + 1),(x - 1, y - 1)]
        neighbors = list(filter(self.isCellInMap, neighbors))
        return neighbors

    def output_agent_map(self):
        #print("Cells: ", self.cells)
        print(" ------------- AGENTS MAP ------------- ")
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                print("| ", self.agent_map[x][y], end="")
            print("|", end="")
            print()
        print(" ------------- END OF MAP ------------- ")
