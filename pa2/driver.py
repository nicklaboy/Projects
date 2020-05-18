from agent import Agent
from basicAgent import BasicAgent
dimensions = 10
mines = 20
startingCoordinate = (2,4)
trials = 50
conduct_trials = trials
improved = 0
basic = 0
while(conduct_trials > 0):

    improved_performance = Agent(dimensions, mines, startingCoordinate, -1)

    basic_performance = BasicAgent(dimensions,mines,startingCoordinate, improved_performance.hidden_map)
    if improved_performance.agent_died < basic_performance.agent_died:
        improved = improved + 1
    elif(improved_performance.agent_died > basic_performance.agent_died):
        basic = basic + 1
    else:
        improved = improved + 1
        basic = basic + 1
    conduct_trials = conduct_trials - 1

improved_wins = improved/trials
basic_wins = basic/trials
print("Improved agent won", improved_wins, "while Basic agent won",basic_wins)
print("Therefore, ", end="")
if(improved_wins > basic_wins):
    print("Improved Won!")
else:
    print("Basic Won!")