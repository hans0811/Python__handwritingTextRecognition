# handwritingTextRecognition

In this project, I used ENMIST and IAM datasets to train two models with nerual network.
The ENMIST was applied to the first model, because the database only had separate letters.
We need use Optical Character Recognition library, such as openCV.

First model with CNN and openCV:
![image](https://github.com/hans0811/handwritingTextRecognition/blob/master/text_CNN_all_test.jpg)

I used more compliated structure on the second model, it involved CNN, LSTM and CTC.

Second model with CNN+LSTM+CTC:
![image](https://github.com/hans0811/handwritingTextRecognition/blob/master/text_cnnlstmctc_all_test.jpg)


CNN+LSTM+CTC structure credit by Kang & Atul
