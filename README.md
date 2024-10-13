# **EECS-498-007-598-005**
Deep Learning for Computer Vision -  Fall 2020

## Assignment Notes
- A1 : KNN
    - split the dataset into 3 parts for train, validation and test. Choose hyperparameters on train and val. Try on test set only once in the end. This is to prevent the model polluted by the test set.
- A2 : linear classifier, two layer network
    - ### SVM
        <div align= "center">
        <img src ="readme_image/weight_matrix.png" height = 50% width = 50%>
        </div>
        
        - The matrix dW represents the gradient (derivative) of the loss with respect to the weight matrix W.
       - $$loss_i = \frac{1}{N}\max(0, \text{scores}[j]_{j \neq y[i]} - \text{scores}[y[i]])$$
         
        for each wrong score of the class that contribute to the loss, the score of the correct class is all subtracted, so dw need to subtract x[i] everytime fingding a wrong score
    - ### Softmax
        - normalize the scores of each class, turning it into probabilities
        - $$P(y = j \mid \mathbf{x}) = \frac{e^{s_j}}{\sum_{k=1}^C e^{s_k}}$$
          $$L = \frac{1}{N}L_i = \frac{1}{N} \sum_{i=1}^N -\log(P(Y = y_i \mid x_i))$$
        - in the code:
            $$\log\left(\frac{e^{s_j}}{\sum_{k=1}^C e^{s_k}}\right) = s_j - \log\left(\sum_{k=1}^C e^{s_k}\right)$$
            $$\frac{\partial L}{\partial W[:,j]} = X[i] \times (prob_j - true\_label_j)$$
    - ### Two_layer_network
        - #### layer1
        $$z_1 = X W_1 + b_1 \quad \text{(pre-activation)}$$
      
        $$h_1 = \text{ReLU}(z_1) \quad \text{(activation using ReLU)}$$
        - #### layer2
        $$z_2 = h_1 W_2 + b_2 \quad \text{(final scores for each class)}$$
    - Nonelinear activation function: do the non-linear work and also preserve the gradient work
    - ### Backpropagation
        - Getting the gradient
        - Getting higher-order derivative 
- A3 : fully connected layer
    - Matrix calculus : why the derivative need to be transposed
    $$out_{ij} = \sum_{k} X_{ik} W_{kj}$$
    $$\frac{\partial L}{\partial X} = \frac{\partial \text{out}}{\partial X} \cdot \frac{\partial L}{\partial \text{out}}$$
    - Modular forward and backward function
    - Update rules for Weight Matrix : SGD, SGD + Momentum, RMSProp, Adam

## Lecture Notes
- ### Data processing : for better process
    - pre-processing
- ### Weight init : Xavier init : for gradient to behave well
    - ReLU correction
    - ResNet
- ### Regularization : prevent overfitting
    - Dropout : large FC layers(many hyperparameters)
    - batch norm, L2 : always works
    - random crop ,scale, cutup, mixup : small dataset
    - ......
- ### Learning Rate Schedule : Start with higer learning rate and decay, complicate optimizer recommend constant
    - Step Schedule
    - Cosine, linear
    - constant
- ### Choosing Hyperparameters
    - Grid Search, Random search(better)
    - tensor board to plot
    - track weight update / weight magnitude(=0.01 about okay)
    - #### without access to many GPUs:
        - build a right model by checking initial loss(if that's correct for random hyperparameters), overfit a small example, find learning rate make loss go down quickly, corase gird for 1~5 epochs, refine grid and train longer, and change hp by looking at learning curves(loss/iteration, train and val accuracy), and again refine grid
- ### After training:
    - train multiple independent models(or have different check points in 1 model) and average the test result (2% extra)
    - Ployak averaging: or keep a moving average of parameters during training and use it for the test time
- ### Transfer learning: models on imageNet tends to also work well on other dataset 
    - use other dataset to extract features then apply model to another dataset to train (use CNN as featrue extractor)
    - Fine-tuning for larger dataset : continue training CNN after feature extracted and get new layers on top of that, tricks: train with feature extration before fine-tuning, lower LR, freeze lower layer save the computation
- ### Distributed training : tons of GPUs
    - data parallelism: split data

        
