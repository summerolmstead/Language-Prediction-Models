#Summer Olmstead , ML Assignment 1

#first import packages 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
#need to use random sampling
import random


#first we will open the files and read the data
#first english file
english_file = open("english.txt")
english_lines = english_file.readlines()
#this showcases the first 10 lines from file and notice /n issue
#print(english_lines[:10])
#this corrects it by just essentially getting rid of the /n
#print([s.replace('\n', '') for s in english_lines[:10]])
#use these principles in the code function to order the data in training and test seperation
training_dataset = []
target_dataset = []
#recall we are only using 5 word length
#define function and search ever word in function
for word in english_lines:
    # Clean the line by removing the new-line character at end of each word
    cleaned_word = word.replace('\n', '')
    # Check if the length of the cleaned word is equal to 5, to get words with 5 characters.
    if len(cleaned_word) == 5:
        # Make an array for converting word to ord representation
        word_to_ord = []
        # Iterate through the cleaned word characters, ord the character, and append it to the word_to_ord list.
        for char in cleaned_word:
            word_to_ord.append(ord(char))
        # Append the ord'ed word to the training dataset
        training_dataset.append(word_to_ord)
        # Append the correct answer to the target dataset, for english we use 0 
        target_dataset.append(0)
#lets visually see results for the first words in training data to be in list(first 10)
#print("First 10 training data:")
#for word in training_dataset[:10]: print(word)
#notice they are all 0 which is good bc this is english
#print(f"\nFirst 10 target data: \n{target_dataset[:10]}")

#good practice to close file when done using
english_file.close()
#lets now do german file ! but add to same sets
german_file = open("german.txt")
german_lines = german_file.readlines()

#func for german to append data, recall prev comments for each line displayed in english loop
for word in german_lines:
    cleaned_word = word.replace('\n', '')
    if len(cleaned_word) == 5:
        word_to_ord = []
        for char in cleaned_word:
            word_to_ord.append(ord(char))
        training_dataset.append(word_to_ord)
        #for german we use 1
        target_dataset.append(1)
german_file.close()

#repeat for last french file
french_file = open("french.txt")
french_lines = french_file.readlines()

#func for french to append data
for word in french_lines:
    cleaned_word = word.replace('\n', '')
    if len(cleaned_word) == 5:
        word_to_ord = []
        for char in cleaned_word:
            word_to_ord.append(ord(char))
        training_dataset.append(word_to_ord)
        #for french we use 2
        target_dataset.append(2)
french_file.close()

# ok now we have our large training(x) and target(y, label) datasets
#we need 20% of the data we just put into the training data set to be into the test data set
#lets randomly select .2*len(training_dataset) amount of elements to append to testing_dataset
#also doing this bc its easier bc data with requirements(len=5) and converted to ord already done 
percent_20_words = int(.2 * len(training_dataset))

#now lets make a function to pick up to this amount to append to training set and delete element from training
testing_dataset = []
target_testing_ds = []

#randomly select words from our training dataset and move them to the target dataset
#do this bc we already have complied all of the training datset into 5 letter words 
for i in range(percent_20_words): #only need to do this 20% of total
    #index needs to match original list and target list and random
    index = random.randrange(len(training_dataset))
    #we get the elements for random list and simultaneously pop it off training list
    random_words = training_dataset.pop(index)
    #get the corresponding target index and pop it off too while getting matching one
    corresponding_rand_target = target_dataset.pop(index)
    #now we add the random word and target elms to new respective testing and target testing dataset
    testing_dataset.append(random_words)
    target_testing_ds .append(corresponding_rand_target)

#test to make sure this is random
#print("First 10 testing data:")
#for word in testing_dataset[10:]: print(word)
#print(f"\nFirst 10 target data: \n{target_testing_ds[:10]}") 
#perfect from this above commented line we notice it is random from languages 0,1,2 which is great 
#to show that is truly random and has correctly worked. Note: every time it runs the testing data will be different

#now we make models and compare
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()
#train models
knn_model.fit(training_dataset, target_dataset)
svm_model.fit(training_dataset, target_dataset)
mlp_nn.fit(training_dataset, target_dataset)
#okay now lets predict with out testing data and see what happens
#question: how to test all of testing data ? in a loop but how to verify results on accuracy? 
#now lets make predictions for each model
knn_predictions = knn_model.predict(testing_dataset)
svm_predictions = svm_model.predict(testing_dataset)
mlp_nn_predictions = mlp_nn.predict(testing_dataset)
#accuracy - the mean of times the prediction equals the correct label , commented output from one run
knn_accuracy = np.mean(knn_predictions == target_testing_ds)
print("KNN Accuracy:", knn_accuracy) #68.46% 
svm_accuracy = np.mean(svm_predictions == target_testing_ds)
print("SVM Accuracy:", svm_accuracy)#71.46%
mlp_accuracy = np.mean(mlp_nn_predictions == target_testing_ds)
print("MLP Accuracy:", mlp_accuracy)#69.55%

#okay generally the accuracy seems to be around 68-71% for all models lolol 
#svm appears to always be the highest one
#now need to figure out deductions and which is the worst performing model
if knn_accuracy < svm_accuracy:
    worst_performing = knn_accuracy
    worst_perform_name = "KNN"
if svm_accuracy < knn_accuracy:
    worst_performing = svm_accuracy
    worst_perform_name = "SVM"
if knn_accuracy < mlp_accuracy: #note dont need to check mlp with svm bc already compared svm with knn which is already checked w knn
    worst_performing = knn_accuracy
    worst_perform_name = "KNN"
if mlp_accuracy < knn_accuracy:
    worst_performing = mlp_accuracy
    worst_perform_name = "MLP"

#ok now lets do the deductions from whichever is the worst performing
deductions = 0
#notice this will only deduct points if it is under or equal to 65%
if worst_performing <= 0.65:
    deductions += 1
#this will be the new value reflected in the bar graph
new_worst_perform = worst_performing - deductions 

#now we will convert the results into percentages to put as our values
if knn_accuracy == worst_performing:
    final_knn = new_worst_perform*100
    final_svm = svm_accuracy*100
    final_mlp = mlp_accuracy*100
if svm_accuracy == worst_performing:
    final_knn = knn_accuracy*100
    final_svm = new_worst_perform*100
    final_mlp = mlp_accuracy*100
if mlp_accuracy == worst_performing:
    final_knn = knn_accuracy*100
    final_svm = svm_accuracy*100
    final_mlp = new_worst_perform*100

final_worst = new_worst_perform*100
print("Worst Performance model:",worst_perform_name,"with an accuracy percentage of: ", round(final_worst,2)), 


labels = ("KNN", "SVM", "MLP")
# Numbers that you want the bars to represent, also we will round to 2 decimals
value = [round(final_knn,2),round(final_svm,2),round(final_mlp,2)]
# Title of the plot
plt.title("Model Accuracy")
# Label for the x values of the bar graph
plt.xlabel("Accuracy")
# Drawing the bar graph
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)
# Display the graph
plt.show()





