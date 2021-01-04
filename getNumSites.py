# Method to get the number of atoms in a computem xyz file
# numLines also corresponds to the total number of atoms
def getNumLines(target):
    numLines = -3
    with open(target, 'r') as inputFile:
        for line in inputFile:
            numLines += 1
    return numLines


# Method to get the total number of possible defect sites in an xyz file
# Still needs more work
def getNumSites(target):
    numMo = 0
    numS = 0

    with open(target, 'r') as inputFile:

        lineTracker  = -2
        S_layer = lineTracker%3

        for line in inputFile:
            if line[0:2]=='42':
                numMo += 1
            elif line[0:2]=='16':
                if S_layer==1:
                    numS += 1
                else:
                    pass
            lineTracker += 1

    numSites = numMo + numS
    return numSites


# Can use the number of sites to generate a concentration of defects that matches the defect concentration specified by the user.
# Probably need to modify this part to call xyz files in different directories
# without the user having to specify the full file path.
if __name__ == "__main__":
    targetFile = input("Enter the name of the image xyz file: ")
    numSites = getNumSites(targetFile)
    print("Total number of possible defect sites is: " + str(numSites))
