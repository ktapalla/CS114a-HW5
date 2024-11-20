# README - COSI 114a HW5 

The code provided in this repository contains the solutions to HW5 for COSI 114a - Fundamentals of Natural Language Processing I. The assignment had us implement an averaged perceptron and use it for language identification of news text. 

As this assignment was done for a class, some helper files and testing files were provided. All student-written solutions to the assignment were written in the ``` hw5.py ``` file. 

## Installation and Execution 

Get the files from GitHub and in your terminal/console move into the project folder. To run the test file included with the files given to students, run the following: 

``` bash 
python test_hw5.py 
```

Doing this will run the set of tests in the test file that was provided to students to test their code. Running the test file will print a score to the user's console indicating the success of the program for each given test case. Make sure to unzip the ``` test_data.zip ``` file before running the above code. It has been compressed to save space on the upload, but is necessary for testing the code. 

Note: The test file provided only made up a portion of the final grade for the assignment. More extensive tests were done during final grading, but students weren't given access to those tests. Furthermore, these instructions assume that the user has python downloaded and is able to run the ``` python ``` command in their terminal. 


## Assignment Description 

### CharBigramFeatureExtractor and InstanceCounter 

#### Extracting Features

This assignment calls for classifying over many  possible languages, each with their own unique vocabulary, thus making it impractical to train a model with sufficiently large training data to capture the lexicon of each language - training with token-level features will lead to encountering a lot of features in the dev and test datasets that weren't present in the training data so they will be uninformative. To solve this issue and avoid dealing with irrelevant tokenization, we reduce our feature set size by instead looking at character-level features. Although different language families use different scripts, the total number of character-level bigrams is still far lower than the total number of token-level bigrams. 

This is implemented through the ``` extract_features(self, instance: LanguageIdentificationInstance) -> ClassificationInstance ```, which is the only method in the ``` CharBigramFeatureExtractor ``` class. It works as follows: 

1. The ``` label ``` attribute is the ``` language ``` attribute of the ``` LanguageIdentificationInstance ``` 
2. The ``` features ``` attribute contains the unique character bigrams generated from the ``` text ``` field of the ``` LanguageIdentificationInstance ```. The order of features doesn't matter, and the beginnings and ends of the strings aren't padded when generating the bigrams. 

Note: It is assumed that the value of the ``` text ``` attribute of the ``` ClassificationInstance ``` provided is a string of length two or greater. 

#### Counting Instances

The ``` InstanceCounter ``` is implemented to provide a canonical list of all of the labels in the training data. It is done through the following two methods: 

1. ``` count_instances(self, instances: Iterable[ClassificationInstance]) -> None ``` 
* Used to iterate over the dataset and populate the ``` InstanceCounter ``` data structures as needed 
2. ``` labels(self) -> list[str] ``` 
* Returns a list of the unique labels in the data in the sorted order provided by the ``` items_descending_value ``` helper function provided in the skeleton code, which requires a ``` Counter ``` over all of the labels. The list is created and stored at the end of ``` count_instances ```; this method doesn't call ``` items_descending_value ```, but rather just returns a list stored in ``` self ```. 

### Perceptron Training 

There are two components of this step: completing the provided methods in the ``` Perceptron ``` model class and writing the training loop that is located in a standalone function outside of the model class. 

#### The ``` Perception ``` Class 

Below are the four methods implemented in the ``` Perception ``` class: 

1. ``` __init__(self, labels: list[str]) -> None ``` 
* Initialization of the class where the following data structures have been set up: 
    * ``` self.labels: list[str] ``` - The labels provided to ``` __init__ ``` 
    * ``` self.weights: dict[str, defaultdict[str, float]] ``` - The weights for each label (outer key) and feature (inner key)
    * ``` self.sums: dict[str, defaultdict[str, float]] ``` - The sums needed by averaging, with the same key structure as weights 
    * ``` self.last_updated: dict[str, defaultdict[str, int]] ``` - The "last updated" value needed by averaging, with the same key structure as weights 
2. ``` classify(self, features: Iterable[str]) -> str ```
* Returns the label with the highest sum of the weights for the features given an ``` iterable ``` of features (ex: the tuple of features that's stored in each ``` ClassificationInstance ```). Calls ``` max_item ``` to get the maximum. 
3. ``` learn(self, instance: ClassificationInstance, step: int, lr: float) -> None ```
* This method is calls for each step in the training loop. Given a ``` ClassificationInstance ``` and the time ``` step ``` in the training loop that it's currently at, it updates the model's weights. 
* The learning rate (``` lr ```) is the amount the weights are shifted each time ``` learn ``` is called. Typically has a value of 1.0. 
3. ``` predict(self, test: Sequence[ClassificationInstance]) -> list[str] ``` 
* Essentially a batched version of classify, where instead of getting a single iterable of features, a sequence of ``` ClassificationInstance ``` is provided. Each instance in the sequence has the ``` classify ``` method called on it to do so. 
4. ``` average(self, final_step: int) -> None ```
* Does the averaging needed (discussed more below)

The ``` learn ``` and ``` average ``` methods are the only methods that modify ``` weights ``` and ``` sums ```. Only ``` learn ``` modifies ``` last_updated ```. 

#### Training the Model 

The training algorithm is described as follows: 

``` train_perceptron(model: Perceptron, data: list[ClassificationInstance], epochs: int, lr: float, *, average: bool, ) -> None ``` 

1. Implementation 
* Initializes ``` step ``` to 1 
* Iterates over the data ``` epochs ``` times 
    * In each epoch, the following is done for each instance: 
        * The model's ``` learn ``` method is called on the instance 
        * Increment ``` step ``` by one
    * At the end of the epoch, the data is shuffled using ``` random.shuffle ``` 
2. Parameters 
* ``` model ``` -  An instance of the ``` Perceptron ``` class 
* ``` data ``` - A list of ``` ClassificationInstance ``` instances. This is the training data for the model. Note that ``` data ``` must always be a list to support the random shuffling that happens, which is different from all previous assignments where iterables were used instead of providing lists of data. 
* ``` epochs ``` - The number of times to fully loop over the training data 
* ``` lr ``` - The learning rate 
* ``` average ``` - Whether you want to perform averaging at the end of training. Note that since ``` average ``` is after a ``` * ``` in the arguments list, it must be specified using keywords (ex: ``` average=True ``` or ``` average=False ```) when calling the function. 

### Weight Averaging 

The goal of averaging is to compute the average value of a weight over the entire course of training, across every step. For a single step at the start, all weights are zero, and then for the remaining steps they are set by the perceptron update rule. 

The implementation of the averaging is optimized by only adding to the sum in two different cases: 1) every time a weight changes or 2) at the very end of training (aka when ``` average ``` is called). Below is a description of the management of two key data structures in the ``` Perceptron ``` class that allows us to do so: 

* ``` self.last_updated ``` tracks the last time a weight was changed. For example, if ``` learn ``` is called with a value of 7 for ``` step ``` and the prediction is incorrect, all the weights that change will have their value in ``` last_updated ``` set to 7. Since ``` last_updated ``` is a default dictionary with a default value of zero, if a weight has not ever been set, its value in ``` last_updated ``` will be zero. 
* ``` self.sums ``` tracks the sums of each weight across all time steps. It is lazily updated, meaning it's only changed when a weight is updated. For example, if ``` learn ``` is called with a value of 7 for ``` step ``` and the prediction is incorrect, the weights for the features present in that instance are all updated. Say the weight for "hooray" for the label "positive" is increased from 2.0 to 3.0 at step 7, and it was last updated at step 3. It's been 7-3=4 steps that the weight has had the value of 2.0, so we all 4*2.0=8.0 to ``` sums ``` for "hooray" with the label "positive". ``` last_updated ``` is set to 7 for "hooray" with label "positive" and the weight is changed to 3.0. 

Below is a description of what to do at each step of averaging: 

* All weights start at value zero at step zero. The first prediction when ``` learn ``` is called will be the first label in the list of labels. 
* ``` train_perceptrons ``` is called for one epoch with a specified learning rate 
    * ``` step ``` is initialized to 1 
    * The instances are looped over: 
        * ``` learn ``` is called with a step value of 1. Each feature and its label is taken. A prediction is made and the weights for each possible label for that feature is updated accordingly (the update is determined by the accuracy of the prediction to the actual label). 
            * Before the weights are changed, ``` self.sums ``` is updated to add the weight since the last update to the total. So the sums get (steps since last update *  the current weight) added to them. 
            * The weights for each label for that feature is changed accordingly by the learning rate. 
            * It's noted that the weights were last updated at this step (1)
        * ``` learn ``` is called with step 2. The above portion is repeated until all the necessary steps are completed. 
    * Now that all the data points have been processed, the sums need to be updated all the way to the end of the training. This is done by having ``` train_perceptrons ``` call on the ``` average ``` method on the model with the final value of step. 
        * Like when weights were updates, we add (current_step - last_updated) * weight to the sum for each weight. This reflects the fact that since the last update, each weight has held its value constant until the end of training. 
        * Finally, the sum for each weight is divided by ``` step ``` and the weight is set to that value. For example, if the sum is 24 and the number of steps is 8, then the final weight would be 3. 

The last method to implement in the ``` Perceptron ``` class is ``` average(self, final_step: int) -> None ```. This method is only called once by the training function at the termination of the training loop. The value ``` final_step ``` that is passed into it is simply the total number of steps in the training loop. It is in this function that the final average of each weight is computer by dividing the sums of every tracked weight by ``` final_step ``` and assigning this new value as the weight in the model's weight data structure. 