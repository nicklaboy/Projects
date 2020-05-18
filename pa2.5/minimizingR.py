from generateMineSweeperMap import GenerateMineSweeperMap
from definitionsForAgent import MineSweeper
from random import random, randint

class Constraints:
    def __init__(self, constraint, value):
        self.constraint = constraint
        self.value = value

def random_select(neighbors, covered, p):
    index = randint(0, len(neighbors) - 1)
    if len(covered) > 0 and random() <= p:
        return covered[index]
    else:
        return neighbors[index]

def getConstraintListLength(c):
    return len(c.constraint)

class MinimizingRisk(GenerateMineSweeperMap):
    def __init__(self, dim, mines, startingCoordinate, isMapPassed):
        super().__init__(dim, mines, startingCoordinate, isMapPassed)
        self.dim = 3
        self.map = [[2,'C',1],['C','C','C'],['C',3,'C']]
        self.startingCoordinate = startingCoordinate
        self.flaggedCells = []
        (x,y) = startingCoordinate
        unknown = []
        #self.createMineSweeperMap()
        self.isVisited = {}
        for x_o in range(dim):
            for y_o in range(dim):
                self.isVisited[(x_o, y_o)] = False

        self.constraint_list = []
        self.constraint_list.append(Constraints([self.startingCoordinate], 0))
        self.flagged_cells = []
        self.known_cells = []

    def isCellKnown(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] != MineSweeper.CLEAR and self.agent_map[x][y] != MineSweeper.FLAG

    def isCellUnknown(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] == MineSweeper.CLEAR

    def isCellFlagged(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] == MineSweeper.FLAG

    def isCellInMap(self, coordinate, dim):
        (x, y) = coordinate
        if (0 <= x < dim) and (0 <= y < dim):
            return True
        else:
            return False

    def updateLocalMap(self, coordinate):
        (x, y) = coordinate
        if self.agent_map[x][y] == MineSweeper.FLAG:
            return -1
        else:
            return self.updateAgentMap(coordinate)

    def updateKnownCells(self, safe_cells):
        flagged = []
        for clue in safe_cells:  # Constraint on clues
            (x, y) = clue
            neighbors = self.adjacent_cells_agent_map(clue)
            known_clues = list(filter(self.isCellKnown, neighbors))
            if 8 - len(known_clues) == self.agent_map[x][y]:
                covered_cells = list(filter(self.isCellUnknown, neighbors))
                for mine in covered_cells:
                    (x_f, y_f) = mine
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    flagged.append(mine)
        return flagged

    def update_flagged_constraints(self, coordinate):
        self.constraint_list.sort(key=getConstraintListLength)
        neighbors = self.adjacent_cells_agent_map(coordinate)
        flags = list(filter(self.isCellFlagged, neighbors))
        for flag in flags:
            for c in self.constraint_list:
                if flag in c.constraint:
                    c.constraint.remove(flag)
                    c.value = c.value - 1

    def doesConstraintExist(self, constraint, value):
        self.constraint_list.sort(key=getConstraintListLength)
        for c in self.constraint_list:
            if len(c.constraint) > len(constraint):
                break
            elif set(constraint).issubset(c.constraint) and c.value != value:
                difference = set(constraint) - set(c.constraint)
                c.constraint.extend(difference)
                return True
        return False

    def isClueOrMine_constraint(self):
        found_clues, found_mines = 0, 0
        for c in self.constraint_list:
            remove_c = 0
            if c.value == 0:
                remove_c = 1
                for safe in c.constraint:
                    if safe not in self.known_cells:
                        value = self.updateLocalMap(safe)
                        (x_c, y_c) = safe
                        self.agent_map[x_c][y_c] = value
                        self.known_cells.append(safe)
                        found_clues = found_clues + 1

            elif len(c.constraint) == c.value:
                remove_c = 1
                for mine in c.constraint:
                    if mine not in self.flagged_cells:
                        (x_f, y_f) = mine
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        self.flagged_cells.append(mine)
                        found_mines = found_mines + 1
                        return found_mines
            if remove_c == 1:
                self.constraint_list.remove(c)
        # if (found_clues > 0):
        #     print("Resolved ", found_clues, " clues")
        # if (found_mines > 0):
        #     print("Resolved ", found_mines, " mines")
        return found_clues, found_mines

    def update_constraints(self, coordinate):
        for c in self.constraint_list:
            if coordinate in c.constraint:
                c.constraint.remove(coordinate)
                if self.isCellFlagged(coordinate):
                    c.value = c.value - 1

    def simplify_constraints(self):
        self.constraint_list.sort(key=getConstraintListLength)
        for c1 in self.constraint_list:
            if len(c1.constraint) == 0:
                continue
            for c2 in self.constraint_list:
                if c1 == c2 or len(c2.constraint) == 0:
                    continue
                if set(c1.constraint).issubset(set(c2.constraint)):
                    c2.constraint = list(set(c2.constraint) - set(c1.constraint))
                    c2.value = c2.value - c1.value
                if set(c2.constraint).issubset(set(c1.constraint)):
                    c1.constraint = list(set(c1.constraint) - set(c2.constraint))
                    c1.value = c1.value - c2.value
        self.isClueOrMine_constraint()

    def add_constraints(self, coordinate):

        neighbors = self.adjacent_cells_agent_map(coordinate)
        flags = list(filter(self.isCellFlagged, neighbors))
        unknown_cells = list(filter(self.isCellUnknown, neighbors))

        (x_o, y_o) = coordinate
        constraint_val = self.agent_map[x_o][y_o] - len(flags)

        exists = self.doesConstraintExist(unknown_cells, constraint_val)
        if not exists:
            self.constraint_list.append(Constraints(unknown_cells, constraint_val))
        # self.output_constraint()

    def output_constraint(self):
        for c in self.constraint_list:
            print("List: ", c.constraint, " Value: ", c.value)

    def cells_to_visit(self):
        visit = []
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if not self.isVisited[(x, y)]:
                    visit.append((x, y))
        return visit

    def unknown_cells(self):
        unknown = []
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if self.agent_map[x][y] == MineSweeper.CLEAR:
                    unknown.append((x, y))
        return unknown

    def adjacent_cells_agent_map(self, coordinate):
        (x, y) = coordinate
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                     (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        neighbors = list(filter(neighbors))
        return neighbors

    def solve(self):

        stack = [self.startingCoordinate]
        count, agent_died = 0, 0
        self.known_cells = [self.startingCoordinate]
        self.flagged_cells = []
        path = []

        while len(self.flagged_cells) < self.numberOfMines and len(stack) > 0:

            clue = stack.pop()
            # print("Mines: ",self.flagged_cells)
            # print("Coordinate: ", clue)
            # print("Queue: ", queue)
            (x, y) = clue

            self.add_constraints(clue)
            if not self.isVisited[clue]:
                self.isVisited[clue] = True
                path.append(clue)
            else:
                flagged = self.updateKnownCells(self.known_cells)
                if len(flagged) > 0:
                    self.flagged_cells.extend(flagged)
                    for flag in flagged:
                        self.update_constraints(flag)
                self.simplify_constraints()

            cells_to_uncover = []
            neighbors = self.adjacent_cells_agent_map(clue)
            if self.agent_map[x][y] == 0:  # If zero, uncover all neighbors
                cells_to_uncover = neighbors
            else:
                known_clues = list(filter(self.isCellKnown, neighbors))
                flags = list(filter(self.isCellFlagged, neighbors))
                covered_cells = list(filter(self.isCellUnknown, neighbors))

                if self.agent_map[x][y] - len(flags) == len(covered_cells):
                    for cell in covered_cells:
                        (x_f, y_f) = cell
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        if cell not in self.flagged_cells:
                            self.flagged_cells.append(cell)
                    covered_cells = list(filter(self.isCellUnknown, neighbors))
                    cells_to_uncover.extend(covered_cells)

                if (8 - self.agent_map[x][y]) - len(known_clues) == len(covered_cells):
                    cells_to_uncover.extend(covered_cells)

                if len(known_clues) < 2 and len(cells_to_uncover) == 0 and len(covered_cells) > 1:
                    cells_to_uncover.append(random_select(covered_cells, [], -1))
                elif len(covered_cells) == 1:
                    cells_to_uncover.append(covered_cells[0])

            # self.output_agent_map()
            for cellToUncover in cells_to_uncover:
                clueOrMine = self.updateLocalMap(cellToUncover)
                if clueOrMine == -1:
                    agent_died = agent_died + 1
                    (x_f, y_f) = cellToUncover
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    self.update_constraints(cellToUncover)
                else:
                    (x_o, y_o) = cellToUncover
                    self.agent_map[x_o][y_o] = clueOrMine
                    if not self.isVisited[cellToUncover]:
                        self.known_cells.append(cellToUncover)
                        stack.append(cellToUncover)

            if len(stack) == 0 and len(self.flagged_cells) < self.numberOfMines:
                stack = self.workedOutMines(path, stack)

            else:
                stack = self.forceRestart((path, stack))

            count += 1

        # self.output_agent_map()
        # self.validate_agent_solution(self.agent_map)
        # print("The agent died ", agent_died, "times, but ", len(self.flagged_cells), " mines were found.")

    #######################
    def workedOutMines(self, path, stack):
        q = 0 #from the probability
        temp_path = path
        temp_clue = temp_path.pop()
        R =0
        S =0
        uncover = 0
        final_Total_Risk =0
        greatestCell = 0
        expected_Number_of_Squares =0
        #create the 3x3 matrix for the cells
        neighbors = self.adjacent_cells_agent_map(temp_clue)
        unknown = list(filter(self.isCellUnknown(), neighbors))
        final_Total_Risk = len(unknown)
        while len(temp_path) > 0:
            count =0
            while(len(unknown) == 0 or len (neighbors) == 0 ) and len(temp_path) > 0:
                temp_path = temp_path.pop()
                neighbors = self.adjacent_cells_agent_map(temp_clue)
                unknown = list(filter(self.isCellUnknown(), neighbors))
            if len(unknown) ==0:
                break
            elif len(unknown) == 1:
                uncover = unknown[0]
            else:
                cl = self.isClueOrMine_constraint()
                for mine in cl.constraint:
                    if mine not in self.updateLocalMap(mine):
                        (x_f, y_f) = mine
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        self.flagged_cells.append(mine)
                        found_mines = found_mines + 1
                        R = found_mines
                q = 0
                S = 8-R
                # q = unknown.pop
                # if q == 0:
                #     R = 0 #number of mines that be be worked out
                # if q >= 1 or q < 2:
                #     R = 1
                # q = 1-q
                # if q == 0:#some other number from probability
                #     S =0 # Number of safe cells that can be worked out
            expected_Number_of_Squares[count] = q * R
            hold = (1 - q) * S
            expected_Number_of_Squares[count] = expected_Number_of_Squares + hold
            count += 1
            clueOrMine = self.updateLocalMap(uncover)
            for i in expected_Number_of_Squares:
                if(expected_Number_of_Squares[i] > expected_Number_of_Squares[i-1]):
                    greatestCell = expected_Number_of_Squares[i]
                    final_Total_Risk+=1
                if(expected_Number_of_Squares[i] == expected_Number_of_Squares[i-1]):
                    greatestCell = random_select(unknown, [], -1)
            greatestCell = self.updateLocalMap(uncover)
            (x_o, y_o) = uncover
            self.agent_map[x_o][y_o] = clueOrMine
            if not self.isVisited[uncover]:
                self.known_cells.append(uncover)
                stack.append(uncover)
                break
        print("works")
        return final_Total_Risk

        # print(expected_Number_of_Squares)

        # safeMine =[]
        # queue = [self.startingCoordinate]
        # while(len(self.flaggedCells)< self.numberOfMines and len(queue) > 0):
        #     clue = queue.pop
        #     (x,y) = clue
        #     cellsToUncover = []

    #######################
    def forceRestart(self, path, stack):
        temp_path = path
        temp_clue = temp_path.pop()

        neighbors = self.adjacent_cells_agent_map(temp_clue)
        unknown = list(filter(self.isCellUnknown, neighbors))

        while len(temp_path) > 0:
            while (len(unknown) == 0 or len(neighbors) == 0) and len(temp_path) > 0:
                temp_clue = temp_path.pop()
                neighbors = self.adjacent_cells_agent_map(temp_clue)
                unknown = list(filter(self.isCellUnknown, neighbors))
            if len(unknown) == 0:
                break
            elif len(unknown) == 1:
                uncover = unknown[0]
            else:
                uncover = random_select(unknown, [], -1)
            clueOrMine = self.updateLocalMap(uncover)
            if clueOrMine == -1:
                (x_f, y_f) = uncover
                self.agent_map[x_f][y_f] = MineSweeper.FLAG
                self.isVisited[uncover] = True
                self.update_constraints(uncover)
                unknown.remove(uncover)
                if uncover not in self.flagged_cells:
                    self.flagged_cells.append(uncover)
            else:
                (x_o, y_o) = uncover
                self.agent_map[x_o][y_o] = clueOrMine
                if not self.isVisited[uncover]:
                    self.known_cells.append(uncover)
                    stack.append(uncover)
                    break
        if len(stack) == 0:
            restart = self.unknown_cells()
            pick = self.startingCoordinate
            while len(restart) > 0:
                if len(restart) > 1:
                    pick = random_select(restart, [], -1)
                elif len(restart) == 1:
                    pick = restart[0]
                clueOrMine = self.updateLocalMap(pick)
                if clueOrMine == -1:
                    (x_f, y_f) = pick
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    restart.remove(pick)
                    self.isVisited[pick] = True
                    self.update_constraints(pick)
                    if pick not in self.flagged_cells:
                        self.flagged_cells.append(pick)
                else:
                    (x_o, y_o) = pick
                    self.agent_map[x_o][y_o] = clueOrMine
                    if not self.isVisited[pick]:
                        self.known_cells.append(pick)
                        stack.append(pick)
                        break
        return stack