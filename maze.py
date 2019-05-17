import numpy as np

def get_possNextStates(s, F, ns):
  possNextStates = []
  for j in range(ns):
    if F[s,j] == 1: possNextStates.append(j)
  return possNextStates

def getRndNextState(s, F, ns):
  possNextStates = get_possNextStates(s, F, ns)
  nextState = \
    possNextStates[np.random.randint(0,\
    len(possNextStates))]
  return nextState


def walk(start, goal, Q):
  currentLoc = start
  print(str(currentLoc) + "->", end="")
  while currentLoc != goal:
    next = np.argmax(Q[currentLoc])
    print(str(next) + "->", end="")
    currentLoc = next
  print("done")

def train(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
  for i in range(0,max_epochs):
    currentState = np.random.randint(0,ns)  # random start state

    while(True):
      next_s = getRndNextState(currentState, F, ns)
      poss_next_next_states = \
        get_possNextStates(next_s, F, ns)
      max_Q = -9999.99
      for j in range(len(poss_next_next_states)):
        nn_s = poss_next_next_states[j]
        q = Q[next_s,nn_s]
        if q > max_Q:
          max_Q = q
      # Q = [(1-a) * Q]  +  [a * (rt + (g * maxQ))]
      Q[currentState][next_s] = ((1 - lrn_rate) * Q[currentState] \
        [next_s]) + (lrn_rate * (R[currentState][next_s] + \
        (gamma * max_Q)))

      currentState = next_s
      if currentState == goal: break

def my_print(Q):
  rows = len(Q); cols = len(Q[0])
  print("       0      1      2      3      4      5\
      6      7      8      9      10     11     12\
     13     14")
  for i in range(rows):
    print("%d " % i, end="")
    if i < 10: print(" ", end="")
    for j in range(cols): print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")

def main():
  np.random.seed(1)
  print("Setting up the maze. One moment.")

  F = np.zeros(shape=[15,15], dtype=np.int)  # Feasible
  F[0,1] = 1; F[0,5] = 1; F[1,0] = 1; F[2,3] = 1; F[3,2] = 1
  F[3,4] = 1; F[3,8] = 1; F[4,3] = 1; F[4,9] = 1; F[5,0] = 1
  F[5,6] = 1; F[5,10] = 1; F[6,5] = 1; F[7,8] = 1; F[7,12] = 1
  F[8,3] = 1; F[8,7] = 1; F[9,4] = 1; F[9,14] = 1; F[10,5] = 1
  F[10,11] = 1; F[11,10] = 1; F[11,12] = 1; F[12,7] = 1;
  F[12,11] = 1; F[12,13] = 1; F[13,12] = 1; F[14,14] = 1

  R = np.zeros(shape=[15,15], dtype=np.int)  # Rewards
  R[0,1] = -0.1; R[0,5] = -0.1; R[1,0] = -0.1; R[2,3] = -0.1
  R[3,2] = -0.1; R[3,4] = -0.1; R[3,8] = -0.1; R[4,3] = -0.1
  R[4,9] = -0.1; R[5,0] = -0.1; R[5,6] = -0.1; R[5,10] = -0.1
  R[6,5] = -0.1; R[7,8] = -0.1; R[7,12] = -0.1; R[8,3] = -0.1
  R[8,7] = -0.1; R[9,4] = -0.1; R[9,14] = 10.0; R[10,5] = -0.1
  R[10,11] = -0.1; R[11,10] = -0.1; R[11,12] = -0.1
  R[12,7] = -0.1; R[12,11] = -0.1; R[12,13] = -0.1
  R[13,12] = -0.1; R[14,14] = -0.1

  #Qlearning quality check
  Q = np.zeros(shape=[15,15], dtype=np.float32)

  print("Analyzing maze using Reinforcement Learning")
  print("Algorithm being used is Q-Learning")
  start = 0; goal = 14
  numStates = 15
  gamma = 0.5
  lrn_rate = 0.5
  max_epochs = 2000
  train(F, R, Q, gamma, lrn_rate, goal, numStates, max_epochs)
  print("Finished")

  print("Q Matrix: \n ")
  my_print(Q)

  print("Using Q to go from 0 to 14 (the goal of this map)")
  walk(start, goal, Q)

if __name__ == "__main__":
    main()

#https://visualstudiomagazine.com/articles/2018/10/18/q-learning-with-python.aspx used as a resource
