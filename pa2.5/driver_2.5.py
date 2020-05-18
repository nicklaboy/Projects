from basic_agent import BasicAgent
from basic_agent_pa2 import BasicAgent2

def computePerformanceOfAgent(results):
    totalSum = 0
    for result in results:
        totalSum += result
    avg = totalSum/(len(results))
    return (round(avg,2) * 100)

def computeMineDensityPerformance(results):
    density_result_avg = {}
    for density_, test_results in results.items():
        print(test_results)
        density_result_avg[density_] = computePerformanceOfAgent(test_results)
    return density_result_avg
dimensions = 16
density = 0.05
mineDensityIndex = 0
mines = int((dimensions ** 2) * density)
mine_densities = [density]
numberOfMines = [mines]
startingCoordinate = (1,1)

trials = 100
trialsConducted = 0
improved_res_v2_5 = {}
improved_res_v2_0 = {}
improved_res_v2_5_sub_trial = []
improved_res_v2_0_sub_trial = []
while(trialsConducted < trials):

    mineDensityIndex = len(numberOfMines) - 1
    improved_performance_v2_5 = BasicAgent(dimensions, numberOfMines[mineDensityIndex], startingCoordinate, -1, False)
    #improved_performance_v2_5.output_agent_map()
    i_perf_v2_5 = (numberOfMines[mineDensityIndex] - improved_performance_v2_5.agent_died) / numberOfMines[mineDensityIndex]
    improved_res_v2_5_sub_trial.append(i_perf_v2_5)

    improved_performance_v2_0 = BasicAgent2(dimensions, numberOfMines[mineDensityIndex], startingCoordinate, improved_performance_v2_5.hidden_map)
    #improved_performance_v2_0.output_agent_map()
    #print(numberOfMines[mineDensityIndex], " ",improved_performance_v2_0.agent_died)
    i_perf_v2_0 = (numberOfMines[mineDensityIndex] - improved_performance_v2_0.agent_died) / numberOfMines[
        mineDensityIndex]
    improved_res_v2_0_sub_trial.append(i_perf_v2_0)
    print("Completed trial #%d" % trialsConducted, " with mine density %.2f" % density)
    trialsConducted = trialsConducted + 1
    if trialsConducted % 5 == 0:
        try:
            improved_res_v2_5[density].extend(improved_res_v2_5_sub_trial)
            improved_res_v2_0[density].extend(improved_res_v2_0_sub_trial)
        except KeyError:
            improved_res_v2_5[density] = improved_res_v2_5_sub_trial
            improved_res_v2_0[density] = improved_res_v2_0_sub_trial

        improved_res_v2_5_sub_trial = []
        improved_res_v2_0_sub_trial = []



        density = density + 0.025
        if density > 0.50:
            density = 0.1
        mines = int((dimensions ** 2) * density)
        numberOfMines.append(mines)
# try:
#     improved_res_v2_5[density].extend(improved_res_v2_5_sub_trial)
#     improved_res_v2_0[density].extend(improved_res_v2_0_sub_trial)
# except KeyError:
#     improved_res_v2_5[density] = improved_res_v2_5_sub_trial
#     improved_res_v2_0[density] = improved_res_v2_0_sub_trial
# print(improved_res_v2_0)
# if len(numberOfMines) == 1:
#     improved_avg_v2_5 = computePerformanceOfAgent(improved_res_v2_5)
#     improved_avg_v2_0 = computePerformanceOfAgent(improved_res_v2_0)
#     print("Average Performance of v2.5 : %.2f" % improved_avg_v2_5, end='')
#     print("%")
#
#     print("Average Performance of v2.0 : %.2f" % improved_avg_v2_0, end='')
#     print("%")
# else:
print(improved_res_v2_5)
print(improved_res_v2_0)
improved_avg_v2_5 = computeMineDensityPerformance(improved_res_v2_5)
improved_avg_v2_0 = computeMineDensityPerformance(improved_res_v2_0)
print("Average Performance of v2.5 : ", improved_avg_v2_5)

print("Average Performance of v2.0 : ", improved_avg_v2_0)

#print(mine_densities)
#print(numberOfMines)




