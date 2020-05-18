import itertools

from generateMineSweeperMap import GenerateMineSweeperMap
from definitionsForAgent import MineSweeper
from random import random, randint
import copy

class Constraints:
    def __init__(self, constraint, value):
        self.constraint = constraint
        self.value = value

class TestValue:
    def __init__(self, coordinate, permutationsTestedWith):
        self.coordinate = coordinate
        self.permutationsTestedWith = permutationsTestedWith

class CellType:
    def __init__(self, coordinate, mine, index):
        self.coordinate = coordinate
        self.mine = mine
        self.index = index

def getConstraintListLength(c):
    return len(c.constraint)

def generateBinaryCombinations(binaryPermutations, used):
    config = []
    for v in binaryPermutations:
        if list(v) not in config and list(v) not in used:
            config.append(list(v))
    return config

def generateBinaryList(constraint, value, used):
    possibleBinaryCombinations = [1] * value
    possibleNullCombinations = [0] * (len(constraint) - value)
    binaryList = possibleBinaryCombinations
    binaryList.extend(possibleNullCombinations)
    # print(binaryList)
    permuteBinaryList = list(itertools.permutations(binaryList, len(constraint)))
    # print(permuteBinaryList)
    binaryCombinations = generateBinaryCombinations(permuteBinaryList, used)
    #print("binary combos: ", binaryCombinations)
    return binaryCombinations

def getConstraintListValue(c):
    return c.value

def createCoordinateTest(coordinate):
    return TestValue(coordinate, {})

class BasicAgent(GenerateMineSweeperMap):
    def __init__(self, dimensions, mines, startingCoordinate, isMapPassed, withRisk):
        super().__init__(dimensions, mines, startingCoordinate, isMapPassed)
        self.withRisk = withRisk
        self.createMineSweeperMap()
        #self.print_hidden_map()
        self.startingCoordinate = startingCoordinate
        (x, y) = startingCoordinate
        self.agent_map[x][y] = self.updateLocalMap(startingCoordinate)

        self.isVisited = {}
        for x_o in range(dimensions):
            for y_o in range(dimensions):
                self.isVisited[(x_o, y_o)] = False

        self.constraint_list = []
        self.constraint_list.append(Constraints([self.startingCoordinate], 0))
        self.flagged_cells = []
        self.known_cells = []

        # self.output_agent_map()
        #self.print_hidden_map()
        self.solve()

    def resetToPriorValues(self, constraints, flagged, known, map):
        self.constraint_list = copy.deepcopy(constraints)
        self.flagged_cells = copy.deepcopy(flagged)
        self.known_cells = copy.deepcopy(known)
        self.agent_map = copy.deepcopy(map)

    def copyPriorValues(self):
        originalConstraints = copy.deepcopy(self.constraint_list)
        originalFlaggedCells = copy.deepcopy(self.flagged_cells)
        originalKnownCells = copy.deepcopy(self.known_cells)
        originalMap = copy.deepcopy(self.agent_map)
        return originalConstraints, originalFlaggedCells, originalKnownCells, originalMap

    def findIntersectingCoordinate(self, coordinate):
        currentCoordinateNeighbors = self.adjacent_cells_agent_map(coordinate)
        if len(currentCoordinateNeighbors) == 8:
            return coordinate
        for neighbor in currentCoordinateNeighbors:
            neighbors = self.adjacent_cells_agent_map(neighbor)
            if len(neighbors) == 8:
                return neighbor

    def getAllCoordinateConstraints(self, coordinate):
        neighbors = self.adjacent_cells_agent_map(coordinate)
        coordinate_constraints = []
        for neighbor in neighbors:
            temp_list = [neighbor]
            for c in self.constraint_list:
                if set(temp_list).issubset(set(c.constraint)) and c.value != 0 \
                        and c not in coordinate_constraints:
                    coordinate_constraints.append(c)
        coordinate_constraints.sort(key=getConstraintListValue)
        return coordinate_constraints

    def getCoordinateTests(self, allConfigTests, coordinate):
        for config in allConfigTests:
            for test in config:
                if test.coordinate == coordinate:
                    return test.permutationsTestedWith

    def removeDuplicateList(self):
        uniqueConstraintsList = []
        for check in self.constraint_list:
            if check not in uniqueConstraintsList:
                uniqueConstraintsList.append(check)
        self.constraint_list = copy.deepcopy(uniqueConstraintsList)

    def isConstraintClue(self, coordinate):
        tempCoordinateList = [coordinate]
        for c in self.constraint_list:

            if set(tempCoordinateList).issubset(set(c.constraint)) and c.value == 0:
                #print(set(tempCoordinateList), " | ", set(c.constraint), " | ", c.value)
                return True
        return False

    def isTestFlagNearZero(self, coorinate):
        (x, y) = coorinate
        return self.agent_map[x][y] == 0

    def output_specfic_constraint(self, c):
        return "" + "list: ", c.constraint, "value: ", c.value

    def checkIfSatisfied(self, flagged, clues, constraints):
        for flag in flagged:
            c = [flag]
            constraints.append(Constraints(c, 1))
        for clue in clues:
            c = [clue]
            constraints.append(Constraints(c, 0))
        constraints.sort(key=getConstraintListLength)

        for c1 in constraints:
            if len(c1.constraint) == 0:
                continue
            for c2 in constraints:
                if c1.value < 0 or c2.value < 0:
                    return False, None, None
                if c1 == c2 or len(c2.constraint) == 0:
                    continue
                if set(c1.constraint).issubset(set(c2.constraint)):
                    if c2.value - c1.value < 0:
                        return False, None, None
                    else:
                        c2.constraint = list(set(c2.constraint) - set(c1.constraint))
                        c2.value = c2.value - c1.value
                if set(c2.constraint).issubset(set(c1.constraint)):
                    if c1.value - c2.value < 0:
                        return False, None, None
                    else:
                        c1.constraint = list(set(c1.constraint) - set(c2.constraint))
                        c1.value = c1.value - c2.value
        mines= []
        clues_  = []
        for c in constraints.copy():
            if c.value < 0:
                return False, None, None
        for c in constraints.copy():
            if len(c.constraint) == 0:
                constraints.remove(c)
                continue
            if c.value == 0:
                for clue in c.constraint:
                    if clue not in clues_:
                        clues_.append(clue)
                constraints.remove(c)

            elif len(c.constraint) == c.value:

                for mine in c.constraint:
                    if mine not in mines:
                        mines.append(mine)
                constraints.remove(c)

        return True, mines, clues_

    def subMatrixConstraint(self, knownNeighbors):
        subConstraintList = []
        for known in knownNeighbors:
            (x,y) = known
            neighbors = self.adjacent_cells_agent_map(known)
            unknown = list(filter(self.isCellUnknown, neighbors))
            flagged = list(filter(self.isCellFlagged, neighbors))
            value = self.agent_map[x][y] - len(flagged)
            subConstraintList.append(Constraints(unknown,value))
        return subConstraintList

    def getConstraintsWithUnknowns(self, unknowns):
        constraints_ = []
        for unknown in unknowns:
            for c in self.constraint_list:
                if unknown in c.constraint and c not in constraints_:
                    constraints_.append(c)
        return constraints_

    def findSharedCoordinateInConstraints(self, unknownNeighbors, constraints):
        shared_constraints = {}
        unknownsInAllList = []
        for unknown in unknownNeighbors:
            shared_constraints[unknown] = []
            for c in constraints:
                if unknown in c.constraint:
                    shared_constraints[unknown].append(c)
            if len(shared_constraints[unknown]) == len(constraints):
                unknownsInAllList.append(unknown)
        return shared_constraints, unknownsInAllList

    def indexMapOfList(self, list):
        indexMap = {}
        for count in range(len(list)):
            indexMap[list[count]] = count
        return indexMap

    def minSharedConstraintValue(self, constraints):
        constraints.sort(key=getConstraintListValue)

        if len(constraints) > 0:
            return constraints[0].value
        else:
            return 1

    def testTwo(self, coordinate):
        self.simplify_constraints(True)
        constraints, flagged, known, agent_map = self.copyPriorValues()
        #self.output_agent_map()
        #self.output_constraint()
        #print("####### Testing 2 #######")

        centerCoordinate = self.findIntersectingCoordinate(coordinate)
        neighbors = self.adjacent_cells_agent_map(centerCoordinate)
        knownNeighbors = list(filter(self.isCellKnown, neighbors))
        unknownNeighbors = list(filter(self.isCellUnknown, neighbors))
        flaggedNeighbors = list(filter(self.isCellFlagged, neighbors))
        (x, y) = centerCoordinate
        if self.agent_map[x][y] == MineSweeper.CLEAR:
            unknownNeighbors.append(centerCoordinate)
        elif self.agent_map[x][y] == MineSweeper.FLAG:
            flaggedNeighbors.append(centerCoordinate)
        else:
            knownNeighbors.append(centerCoordinate)
        exhaustive_constraints = self.getConstraintsWithUnknowns(unknownNeighbors)
        #self.output_constraint_list(exhaustive_constraints)
        unknownIndexMap = self.indexMapOfList(unknownNeighbors)

        shared_constraints, shared_unknowns = self.findSharedCoordinateInConstraints(unknownNeighbors,exhaustive_constraints)
        #print("beee", shared_unknowns)
        goodTests = []
        for unknown in unknownNeighbors:
            shared_subset_c = shared_constraints[unknown].copy()
            #self.output_constraint_list(shared_constraints[unknown])
            shared_c, shared_c_unknowns = self.findSharedCoordinateInConstraints(shared_unknowns, shared_subset_c)
            if unknown in shared_unknowns and unknown in shared_c_unknowns:
                shared_unknowns = []

                valueToSatisfy = self.minSharedConstraintValue(shared_c[unknown])
                shared_combos = generateBinaryList(shared_c_unknowns, valueToSatisfy, [])
                #print("shared ", end='')
            else:
                shared_c_unknowns = [unknown]
                shared_combos = generateBinaryList([unknown], 1, [])
                #print(unknown, " not shared ", end='')
            #print( " combo", shared_combos)
            tests = self.testCheck(shared_combos, shared_c_unknowns, shared_subset_c.copy())
            if(len(tests) > 0 and tests not in goodTests):

                goodTests.extend(tests)
            #goodTests.append(tests)
        used = []
        probabaility = {}
        for unknown in unknownNeighbors:
            probabaility[unknown] = 0
        totalTest = 0
        for test in goodTests:
            if test not in used:
                totalTest = totalTest + 1
                for cell in test:
                    try:
                        probabaility[cell] = probabaility[cell] + 1
                    except KeyError:
                        probabaility[cell] = 1
                used.append(test)
        if totalTest == 0:
            totalTest = 1
        for cell, p in probabaility.items():

            probabaility[cell] = probabaility[cell]/totalTest
            if self.withRisk:
                q = probabaility[cell]
                R,S = self.calculateRisk(cell)
                probabaility[cell] = q * R + (1 - q)*S
        #print("Potential: ", goodTests)
        #print("Prob: ", probabaility)
        #print("###### END 2 ######")
        self.resetToPriorValues(constraints, flagged, known, agent_map)
        return probabaility

    def testCheck(self, shared_combos, shared_c_unknowns,shared_subset_c):
        constraints, flagged, known, agent_map = self.copyPriorValues()
        goodTests = []
        for shared_combo in shared_combos:

            cellsFlagged = []
            cellsClue = []
            for i in range(len(shared_combo)):
                if shared_combo[i] == 1:
                    cellsFlagged.append(shared_c_unknowns[i])
                else:
                    cellsClue.append(shared_c_unknowns[i])
            shared_subset_constraint_ = copy.deepcopy(shared_subset_c)

            base_flag = cellsFlagged.copy()
            base_clues = cellsClue.copy()
            for c in shared_subset_constraint_.copy():
                self.resetToPriorValues(constraints, flagged, known, agent_map)
                c_ = [copy.deepcopy(c)]
                isSatisfied, mines_, clues_ = self.checkIfSatisfied(cellsFlagged, [], c_)
                if isSatisfied:
                    goodTests.append(cellsFlagged)
                else:
                    combos = generateBinaryList(c.constraint, c.value, [])
                    for combo in combos:
                        for index in range(len(combo)):
                            if combo[index] == 1:
                                cellsFlagged.append(c.constraint[index])
                            else:
                                cellsClue.append(c.constraint[index])
                        #print("## SUB TEST 2##")
                        # for flag in cellsFlagged:
                        #     (x_f, y_f) = flag
                        #     self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        #self.output_agent_map()

                        #self.resetToPriorValues(constraints, flagged, known, agent_map)
                        isSatisfied_, mines_o,clues_o = self.checkIfSatisfied(cellsFlagged, [], c_)
                        if isSatisfied_:
                            goodTests.append(cellsFlagged)
                        #else:
                            #print("Failed")
                        cellsFlagged = base_flag.copy()
                        cellsClue = base_clues.copy()
        #print("GOod: ", goodTests)
        return goodTests
    def getCoordinateConstraints(self, coordinate):
        (x,y) = coordinate
        if self.agent_map[x][y] >= 0:
            centerCoordinate = self.findIntersectingCoordinate(coordinate)
            neighbors = self.adjacent_cells_agent_map(centerCoordinate)
            sub_list = list(filter(self.isCellUnknown, neighbors))
        else:
            sub_list = [coordinate]
        constraints = []
        for unknown in sub_list:
            sub_list = [unknown]
            for c in self.constraint_list:
                if set(sub_list).issubset(c.constraint) and c not in constraints:
                    constraints.append(c)
        constraints.sort(key=getConstraintListValue)
        return constraints

    def getVariablesToTest(self, constraintsToSatisfy):
        unknownVariablesToSolve = []
        for c in constraintsToSatisfy:
            for cell in c.constraint:
                if cell not in unknownVariablesToSolve and not (cell in self.known_cells or cell in self.flagged_cells):
                    unknownVariablesToSolve.append(cell)
        return unknownVariablesToSolve

    def allCoordinatesInConstraint(self, constraintsToSatisfy):
        vars = []
        for c in constraintsToSatisfy:
            for cell in c.constraint:
                if cell not in vars:
                    vars.append(cell)
        return vars

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
        for safe in safe_cells:
            self.update_constraints(safe)
        return flagged

    def update_flagged_constraints(self, coordinate):
        self.constraint_list.sort(key=getConstraintListLength)
        for c in self.constraint_list:
            if coordinate in c.constraint:
                if c.value - 1 < 0:
                    return False
                c.constraint.remove(coordinate)
                c.value = c.value - 1
        return True

    def doesConstraintShareSet(self, constraint, value):
        self.constraint_list.sort(key=getConstraintListLength)
        for c in self.constraint_list:
            sharedConstraints = list(set(constraint) & set(c.constraint))
            if len(constraint) > len(sharedConstraints) and sharedConstraints:
                difference_c = set(c.constraint) - set(sharedConstraints)
                value_c = c.value - value
                c.constraint = difference_c
                c.value = value_c
                return True
        return False

    def isClueOrMine_constraint(self, isOkayToUpdate):
        found_clues, found_mines = 0, 0
        resolved_mines, resolved_clues = [], []
        # self.output_constraint()
        for c in self.constraint_list:
            if len(c.constraint) == 0:
                self.constraint_list.remove(c)
                continue
            remove_c = 0
            if c.value == 0:
                remove_c = 1
                resolved_clues.extend(c.constraint)
                for safe in c.constraint:
                    if safe not in self.known_cells:
                        (x_c, y_c) = safe
                        if isOkayToUpdate:
                            value = self.updateLocalMap(safe)
                            self.agent_map[x_c][y_c] = value
                            # self.isVisited[safe] = True
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
                        resolved_mines.append(mine)

            if remove_c == 1:
                # self.output_constraint_list(c)
                self.constraint_list.remove(c)

        #if (found_clues > 0):
        #    print("Resolved ", found_clues, " clues")
        #if (found_mines > 0):
        #    print("Resolved ", found_mines, " mines")
        if len(resolved_mines) > 0 and len(resolved_clues) > 0:
            return resolved_mines, resolved_clues
        elif len(resolved_mines) > 0 and len(resolved_clues) == 0:
            return resolved_mines, None
        elif len(resolved_mines) == 0 and len(resolved_clues) > 0:
            return None, resolved_clues
        else:
            return None, None

    def update_constraints(self, coordinate):
        for c in self.constraint_list:
            if coordinate in c.constraint:
                c.constraint.remove(coordinate)
                if self.isCellFlagged(coordinate):
                    c.value = c.value - 1

    def simplify_constraints(self, isOkayToUpdate):
        self.constraint_list.sort(key=getConstraintListLength)
        for cell in self.known_cells:
            for c in self.constraint_list:
                if cell in c.constraint:
                    c.constraint.remove(cell)

        for c1 in self.constraint_list:
            if len(c1.constraint) == 0:
                continue
            for c2 in self.constraint_list:
                if c1 == c2 or len(c2.constraint) == 0:
                    continue
                if set(c1.constraint).issubset(set(c2.constraint)):
                    if c2.value - c1.value < 0:
                        return False, None, None
                    else:
                        c2.constraint = list(set(c2.constraint) - set(c1.constraint))
                        c2.value = c2.value - c1.value
                if set(c2.constraint).issubset(set(c1.constraint)):
                    if c1.value - c2.value < 0:
                        return False, None, None
                    else:
                        c1.constraint = list(set(c1.constraint) - set(c2.constraint))
                        c1.value = c1.value - c2.value
        resolved_mines, resolved_clues = self.isClueOrMine_constraint(isOkayToUpdate)
        if resolved_clues is not None and len(self.constraint_list) > 0:
            for clue in resolved_clues:
                if len(self.constraint_list) > 0:
                    self.add_constraints(clue, None, isOkayToUpdate)
                else:
                    break
        return True, resolved_mines, resolved_clues

    def minConstraintValue(self, knownNeighbors):
        min = -1
        for known in knownNeighbors:
            (x, y) = known
            if min == -1 or self.agent_map[x][y] < min:
                min = self.agent_map[x][y]
        return min

    def add_constraints(self, coordinate, value, simplify):

        neighbors = self.adjacent_cells_agent_map(coordinate)
        flags = list(filter(self.isCellFlagged, neighbors))
        unknown_cells = list(filter(self.isCellUnknown, neighbors))
        known_cells = list(filter(self.isCellKnown, neighbors))
        (x_o, y_o) = coordinate
        if len(unknown_cells) == 0:
            return
        #print(coordinate, " | u: ", unknown_cells)
        if value is None and not self.agent_map[x_o][y_o] >= 0:
            constraint_val = self.minConstraintValue(known_cells)
        elif value is None and self.agent_map[x_o][y_o] >= 0:
            constraint_val = self.agent_map[x_o][y_o] - len(flags)
            self.update_constraints(coordinate)
        else:
            constraint_val = value

        # self.doesConstraintShareSet(unknown_cells,constraint_val)
        self.constraint_list.append(Constraints(unknown_cells, constraint_val))

        self.simplify_constraints(simplify)

    def output_constraint_list(self, constraint):
        for c in constraint:
            print("List: ", c.constraint, " Value: ", c.value)

    def output_constraint(self):
        for c in self.constraint_list:
            print("List: ", c.constraint, " Value: ", c.value)

    def flagCellsAndUpdateConstraints(self):
        flagged = self.updateKnownCells(self.known_cells)
        if len(flagged) > 0:
            self.flagged_cells.extend(flagged)
            for flag in flagged:
                self.update_constraints(flag)

    def random_select(self,neighbors, covered, p,coordinate):
        index = randint(0, len(neighbors) - 1)

        probability = self.testTwo(coordinate)
        min = 2
        min_cell = []
        for cell,p in probability.items():
            self.calculateRisk(cell)
            if min_cell is None:
                min_cell = cell
            if p < min and cell in covered:
                min = p
                min_cell = [cell]
            elif p == min:
                min_cell.append(cell)
        if min_cell is None and len(covered) > 0:
            return covered[index]
        elif len(covered) == 0:
            return neighbors[index]
        else:
            if(len(min_cell) > 1):
                index = randint(0, len(min_cell) - 1)
            else:
                return min_cell[0]
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
            #self.output_constraint()
            #print("Coordinate: ", clue)
            if not self.isVisited[clue]:
                self.add_constraints(clue, None, True)
                self.isVisited[clue] = True
                path.append(clue)
            else:
                self.flagCellsAndUpdateConstraints()
                self.simplify_constraints(True)

            # PossibleConfigurations(clue)
            cells_to_uncover = []
            neighbors = self.adjacent_cells_agent_map(clue)
            isBaseStrategyReduceMines = False
            if self.agent_map[x][y] == 0:  # If zero, uncover all neighbors base strategy #1
                cells_to_uncover = neighbors
            else:
                known_clues = list(filter(self.isCellKnown, neighbors))
                flags = list(filter(self.isCellFlagged, neighbors))
                covered_cells = list(filter(self.isCellUnknown, neighbors))

                if self.agent_map[x][y] - len(flags) == len(covered_cells):  # base strategy #2
                    for cell in covered_cells:
                        (x_f, y_f) = cell
                        self.agent_map[x_f][y_f] = MineSweeper.FLAG
                        if cell not in self.flagged_cells:
                            self.flagged_cells.append(cell)
                    covered_cells = list(filter(self.isCellUnknown, neighbors))
                    cells_to_uncover.extend(covered_cells)
                    isBaseStrategyReduceMines = True
                if (8 - self.agent_map[x][y]) - len(known_clues) == len(covered_cells):  # base strategy #3
                    cells_to_uncover.extend(covered_cells)
                    isBaseStrategyReduceMines = True

                if not isBaseStrategyReduceMines:
                    if len(known_clues) < 2 and len(cells_to_uncover) == 0 and len(covered_cells) > 1:
                        self.simplify_constraints(True)

                        cells_to_uncover.append(self.random_select(covered_cells, [], -1,clue))
                    elif len(covered_cells) == 1:
                        cells_to_uncover.append(covered_cells[0])

            # self.output_agent_map()

            for cellToUncover in cells_to_uncover:
                clueOrMine = self.updateLocalMap(cellToUncover)

                if clueOrMine == -1:
                    #print("F: ", cellToUncover)
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
            #self.output_agent_map()

            if len(stack) == 0 and len(self.flagged_cells) < self.numberOfMines:
                stack = self.forceRestart(path, stack)

            count += 1
        for row in range(self.dimensions):
            for column in range(self.dimensions):
                coordinate = (row,column)
                val = self.updateLocalMap(coordinate)
                if val == -1:
                    self.agent_map[row][column] = MineSweeper.FLAG
                else:
                    self.agent_map[row][column] = val
        # self.output_agent_map()
        #self.validate_agent_solution(self.agent_map)
        # print("The agent died ", agent_died, "times, but ", len(self.flagged_cells), " mines were found.")

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
                uncover = self.random_select(unknown, [], -1, unknown[0])
            clueOrMine = self.updateLocalMap(uncover)
            if clueOrMine == -1:
                (x_f, y_f) = uncover
                self.agent_map[x_f][y_f] = MineSweeper.FLAG
                # self.isVisited[uncover] = True
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
                    pick = self.random_select(restart, [], -1,restart[0])
                elif len(restart) == 1:
                    pick = restart[0]
                clueOrMine = self.updateLocalMap(pick)
                if clueOrMine == -1:
                    (x_f, y_f) = pick
                    self.agent_map[x_f][y_f] = MineSweeper.FLAG
                    restart.remove(pick)
                    # self.isVisited[pick] = True
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

    def isCellKnown(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] != MineSweeper.CLEAR and self.agent_map[x][y] != MineSweeper.FLAG

    def isCellUnknown(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] == MineSweeper.CLEAR

    def isCellFlagged(self, coordinate):
        (x, y) = coordinate
        return self.agent_map[x][y] == MineSweeper.FLAG

    def adjacent_cells_agent_map(self, coordinate):
        (x, y) = coordinate  #
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                     (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
        neighbors = list(filter(self.isCellInMap, neighbors))
        return neighbors

    def updateLocalMap(self, coordinate):
        (x, y) = coordinate
        if self.agent_map[x][y] == MineSweeper.FLAG:
            return -1
        else:
            return self.updateAgentMap(coordinate)

    def calculateRisk(self, cell):
        #print("Cell r:" , cell)
        constraints, flagged, known, agent_map = self.copyPriorValues()
        isSatisfied, mines, clue = self.checkIfSatisfied([cell], [], self.constraint_list.copy())
        R = 1
        if isSatisfied:
            R = len(mines) + len(clue)
        self.resetToPriorValues(constraints, flagged, known, agent_map)
        isSatisfied_, mines_, clues_ = self.checkIfSatisfied([], [cell], self.constraint_list.copy())
        S = 1
        if isSatisfied_:
            S = len(mines_) + len(clues_)
        self.resetToPriorValues(constraints, flagged, known, agent_map)
        return R, S

    def output_agent_map(self):
        print(" ------------- AGENTS MAP ------------- ")
        for x in range(self.dimensions):
            for y in range(self.dimensions):
                print("| ", self.agent_map[x][y], end="")
            print("|", end="")
            print()
        print(" ------------- END OF MAP ------------- ")
