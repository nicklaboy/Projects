from generateMineSweeperMap import GenerateMineSweeperMap
from definitionsForAgent import MineSweeper
from random import random, randint

class Constraints:
    def __init__(self, constraint, value):
        self.constraint = constraint
        self.value = value

class Agent(GenerateMineSweeperMap):
    def __init__(self, dimensions, mines,startingCoordinate, map):
        super().__init__(dimensions, mines, startingCoordinate,map)
        self.createMineSweeperMap()
        #self.print_hidden_map()

        self.isVisited = {}

        self.startingCoordinate = startingCoordinate
        (x,y) = startingCoordinate
        self.agent_map[x][y] = self.updateAgentMap(startingCoordinate)

        for x_o in range(dimensions):
            for y_o in range(dimensions):
                self.isVisited[(x_o,y_o)] = False

        self.constraint_list = []
        self.constraint_list.append(Constraints([self.startingCoordinate],0))
        self.flagged_cells = []
        self.known_cells = []
        #self.output_agent_map()
        self.solve()

    def isCellKnown(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] != MineSweeper.CLEAR and self.agent_map[x][y] != MineSweeper.FLAG
    def isCellUnknown(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] == MineSweeper.CLEAR
    def isCellFlagged(self, coordinate):
        (x,y) = coordinate
        return self.agent_map[x][y] == MineSweeper.FLAG
    def adjacent_cells_agent_map(self, coordinate):
        (x, y) = coordinate #
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                     (x + 1, y + 1),(x + 1, y - 1),(x - 1, y + 1),(x - 1, y - 1)]
        neighbors = list(filter(self.isCellInMap, neighbors))
        return neighbors


    def random_select(self,neighbors,covered, p):
        index = randint(0, len(neighbors)-1)
        if(len(covered) > 0 and random() <= p):
            return covered[index]
        else:
            return neighbors[index]

    def updateKnownCells(self,safe_cells):
        flagged = []
        #Constraint on clues
        for clue in safe_cells:
            (x,y) = clue
            neighbors = self.adjacent_cells_agent_map(clue)
            known_clues = list(filter(self.isCellKnown, neighbors))
            if(8 - len(known_clues) == self.agent_map[x][y]):
                covered_cells = list(filter(self.isCellUnknown, neighbors))
                for mine in covered_cells:
                    (x_f,y_f) = mine
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG

                    flagged.append(mine)
        return flagged

    def subset_agent_map(self, coordinate):
        (x, y) = coordinate #
        neighbors = [(x - 1, y - 1),(x - 1, y), (x - 1, y + 1),
                     (x, y - 1),(x,y),(x, y + 1),
                     (x + 1, y - 1),(x + 1, y),(x + 1, y + 1)]
        subset = [[]]
        row = 0
        column = 0
        for cell in neighbors:
            (x_o,y_o) = cell
            if (column % 3 == 0):
                row = row + 1
            subset[row][column % 3] = self.agent_map[x_o][y_o]
            column = column + 1
        return subset
    def getConstraintListLength(self, c):
        return len(c.constraint)

    def update_flagged_constraints(self, coordinate):
        self.constraint_list.sort(key=self.getConstraintListLength)
        neighbors = self.adjacent_cells_agent_map(coordinate)
        flags = list(filter(self.isCellFlagged, neighbors))
        for flag in flags:
            for c in self.constraint_list:
                if(flag in c.constraint):
                    c.constraint.remove(flag)
                    c.value = c.value - 1

    def doesConstraintExist(self, constraint, value):
        self.constraint_list.sort(key=self.getConstraintListLength)
        for c in self.constraint_list:
            if(len(c.constraint) > len(constraint)):
                break
            elif(set(constraint).issubset(c.constraint) and c.value != value):
                difference = set(constraint) - set(c.constraint)
                c.constraint.extend(difference)
                return True, None
        return False, None

    def isClueOrMine_constraint(self):
        found_clues, found_mines = 0, 0
        for c in self.constraint_list:
            remove_c = 0
            if (c.value == 0):
                remove_c = 1
                for safe in c.constraint:
                    if (safe not in self.known_cells):
                        value = self.updateAgentMap(safe)
                        (x_c, y_c) = safe
                        self.agent_map[x_c][y_c] = value
                        self.known_cells.append(safe)
                        found_clues = found_clues + 1

            elif (len(c.constraint) == c.value):
                remove_c = 1
                for mine in c.constraint:
                    if (mine not in self.flagged_cells):
                        (x_f, y_f) = mine
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        self.flagged_cells.append(mine)
                        found_mines = found_mines + 1
            if (remove_c == 1):
                self.constraint_list.remove(c)
        # if (found_clues > 0):
        #     print("Resolved ", found_clues, " clues")
        # if (found_mines > 0):
        #     print("Resolved ", found_mines, " mines")
        return found_clues, found_mines

    def update_constraints(self, coordinate):
        for c in self.constraint_list:
            if(coordinate in c.constraint):
                c.constraint.remove(coordinate)
                if(self.isCellFlagged(coordinate)):
                    c.value = c.value - 1
    def simplify_constraints(self):
        self.constraint_list.sort(key=self.getConstraintListLength)
        for c1 in self.constraint_list:
            if(len(c1.constraint)==0):
                continue
            for c2 in self.constraint_list:
                if(c1 == c2 or len(c2.constraint) == 0):
                    continue
                if(set(c1.constraint).issubset(set(c2.constraint))):
                    c2.constraint = list(set(c2.constraint) - set(c1.constraint))
                    c2.value = c2.value - c1.value
                if (set(c2.constraint).issubset(set(c1.constraint))):
                    c1.constraint = list(set(c1.constraint) - set(c2.constraint))
                    c1.value = c1.value - c2.value
        self.isClueOrMine_constraint()

    def add_constraints(self, coordinate):
        (x_o, y_o) = coordinate

        neighbors = self.adjacent_cells_agent_map(coordinate)
        flags = list(filter(self.isCellFlagged, neighbors))
        unknown_cells = list(filter(self.isCellUnknown, neighbors))
        constraint_val = self.agent_map[x_o][y_o] - len(flags)
        exists,update_constraint = self.doesConstraintExist(unknown_cells,constraint_val)
        if(not exists and update_constraint == None):
            #self.output_constraint()
            self.constraint_list.append(Constraints(unknown_cells,constraint_val))
        elif(exists and  update_constraint ==  None):
            self.constraint_list.append(Constraints(unknown_cells,constraint_val))
        self.simplify_constraints()

    def output_constraint(self):
        for c in self.constraint_list:
            print("List: ", c.constraint, " Value: ", c.value)
    def cells_to_visit(self):
        visit = []
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if(self.isVisited[(x,y)] == False):
                    visit.append((x,y))
        return visit
    def unknown_cells(self):
        unknown = []
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                if (self.agent_map[x][y] == MineSweeper.CLEAR):
                    unknown.append((x, y))
        return unknown
    def solve(self):
        queue = [self.startingCoordinate]
        count = 0
        agent_died = 0
        self.known_cells = [self.startingCoordinate]
        unknown_cells = []
        self.flagged_cells = []
        for row in range(self.dimensions):
            for column in range(self.dimensions):
                if((row,column) != self.startingCoordinate):
                    unknown_cells.append((row,column))
        path = []
        while(len(self.flagged_cells) < self.numberOfMines and len(queue) > 0):
            clue = queue.pop()

            #print("Mines: ",self.flagged_cells)
            #print("Coordinate: ", clue)
            #print("Queue: ", queue)

            (x,y) = clue
            cells_to_uncover = []
            self.add_constraints(clue)
            if (self.isVisited[clue] == False):
                path.append(clue)
                self.isVisited[clue] = True
            else:
                self.simplify_constraints()
            neighbors = self.adjacent_cells_agent_map(clue)

            if(self.agent_map[x][y] == 0):# If zero, uncover all neighbors
                cells_to_uncover = neighbors
            else:
                known_clues = list(filter(self.isCellKnown, neighbors))
                flags = list(filter(self.isCellFlagged,neighbors))
                covered_cells = list(filter(self.isCellUnknown,neighbors))

                if(self.agent_map[x][y] - len(flags) == len(covered_cells)):
                    for cell in covered_cells:
                        (x_f, y_f) = cell
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        if (cell not in self.flagged_cells):
                            self.flagged_cells.append(cell)
                    covered_cells = list(filter(self.isCellUnknown, neighbors))

                if ((8 - self.agent_map[x][y]) - len(known_clues) == len(covered_cells)):
                    cells_to_uncover.extend(covered_cells)
                if(len(known_clues) < 2 and len(cells_to_uncover) == 0 and len(covered_cells) > 1):
                    cells_to_uncover.append(self.random_select(covered_cells, [], -1))
                elif(len(covered_cells) == 1):
                    cells_to_uncover.append(covered_cells[0])
                elif(len(known_clues) >= 2):
                    flagged = self.updateKnownCells(self.known_cells)
                    if(len(flagged) > 0):
                        self.flagged_cells.extend(flagged)
                        for flag in flagged:
                            self.update_constraints(flag)
                    self.simplify_constraints()

            #self.output_agent_map()
            for cellToUncover in cells_to_uncover:
                clueOrMine = self.updateAgentMap(cellToUncover)
                if(clueOrMine == -1):
                    agent_died = agent_died + 1
                    (x_f, y_f) = cellToUncover
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    self.update_constraints(cellToUncover)
                else:
                    (x_o, y_o) = cellToUncover
                    self.agent_map[x_o][y_o] = clueOrMine
                    if(self.isVisited[cellToUncover] == False):
                        self.known_cells.append(cellToUncover)
                        queue.append(cellToUncover)

            if(len(queue) == 0 and len(self.flagged_cells) < self.numberOfMines):
                temp_path = path

                temp_clue = temp_path.pop()
                neighbors = self.adjacent_cells_agent_map(temp_clue)
                unknown = list(filter(self.isCellUnknown, neighbors))
                #print(temp_path)
                while(len(temp_path) > 0):
                    while((len(unknown) == 0 or len(neighbors) == 0) and len(temp_path) > 0):
                        temp_clue = temp_path.pop()
                        neighbors = self.adjacent_cells_agent_map(temp_clue)
                        unknown = list(filter(self.isCellUnknown, neighbors))
                    if(len(unknown) == 0):
                        break
                    elif(len(unknown) == 1):
                        uncover = unknown[0]
                    else:
                        uncover = self.random_select(unknown, [], -1)
                    clueOrMine = self.updateAgentMap(uncover)
                    if(clueOrMine == -1):
                        (x_f, y_f) = uncover
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        self.isVisited[uncover] = True
                        self.update_constraints(uncover)
                        unknown.remove(uncover)
                        if (uncover not in self.flagged_cells):
                            self.flagged_cells.append(uncover)
                    else:
                        (x_o, y_o) = uncover
                        self.agent_map[x_o][y_o] = clueOrMine
                        if (self.isVisited[uncover] == False):
                            self.known_cells.append(uncover)
                            queue.append(uncover)
                            break
                if(len(queue) == 0):
                    restart = self.unknown_cells()
                    pick = self.startingCoordinate
                    while(len(restart) > 0):
                        if (len(restart) > 1):
                            pick = self.random_select(restart, [], -1)
                        elif(len(restart) == 1):
                            pick = restart[0]
                        clueOrMine = self.updateAgentMap(pick)
                        if (clueOrMine == -1):
                            (x_f, y_f) = pick
                            self.agent_map[x_f][y_f] = MineSweeper.FLAG
                            restart.remove(pick)
                            self.isVisited[pick] = True
                            self.update_constraints(pick)
                            if(pick not in self.flagged_cells):
                                self.flagged_cells.append(pick)
                        else:
                            (x_o, y_o) = pick
                            self.agent_map[x_o][y_o] = clueOrMine
                            if (self.isVisited[pick] == False):
                                self.known_cells.append(pick)
                                queue.append(pick)
                                break

            count += 1
        if(len(self.flagged_cells) == self.numberOfMines):
            reveal_last_cells = self.unknown_cells()
            for cell in reveal_last_cells:
                (x,y) = cell
                self.agent_map[x][y] = self.updateAgentMap(cell)
        #self.output_agent_map()
        self.validate_agent_solution(self.agent_map)
        #print("The agent died ", agent_died, "times, but ", len(self.flagged_cells), " mines were found.")

    def output_agent_map(self):
        print(" ------------- AGENTS MAP ------------- ")
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                print("| ", self.agent_map[x][y], end="")
            print("|", end="")
            print()
        print(" ------------- END OF MAP ------------- ")