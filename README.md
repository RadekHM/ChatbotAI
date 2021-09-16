# ChatbotAI

Tom, the chatbot is a console based chatbot designed to help costumers talk to an AI, immediately get answers for questions they have and at the same time the ability to maintain basic conversation.
This application is going to make costumer support a more automatic process that does not require any human interaction. The application has a user-friendly console interaction with the ability to quit anytime.
- Admin Section (Can edit the default responses pre training)
- User Section (Can interact with the bot)


## Requiremnets

 1. Code Editor
 2. Python Package Installed

# Libraries Used

- <b>PyTorch</b>: PyTorch is an optimized tensor library for deep learning, I used PyTorch instead of TensorFlow for it’s lightweight, and very low compile time.
- <b>NLTK</b>: NLTK is a leading platform for building python programs to work with human language data, use it mainly for tokenizing and stemming the input array.
- <b>NumPy</b>: NumPy is a python library used for working with arrays


## Training

The chatbot is trained from the json file “intents.json”, it is the main training data, easy to understand and add/remove features as needed.
Whenever the user inputs a question, the bot tries to classify the text with the corresponding tag and gives an answer from the pool of responses.

![](/Images/trainingdata.png)
 
After the training is complete, the program dumps a data.pth file, that is then used for prediction and running the application.
 
 
## Interface

The interface is console based, so the conversation goes fully through the console, where the bot tries to identify what he is being asked, and respond based on prediction.

![](/Images/interface.png)


## Built With

- Python Programing Language


## TODO

- Optimize Code
- Increase responses to cover a wider variety of questions
- Add a web based interface


## Authors

- Radvan Khammud

