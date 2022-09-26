import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy
np.random.seed(1)

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10] 
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Settings for task 3. Keep all to false for task 2.
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    
    X_train, Y_train, *_ = utils.load_full_mnist()
    mean_pixel = np.mean(X_train)           # mean = 33.5527
    std = np.sqrt(np.cov(X_train.flatten()))# std = 78.8755
    
    X_train = pre_process_images(X_train,mean_pixel, std)
    X_val = pre_process_images(X_val, mean_pixel, std) #same std and mean
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_new, val_history_new = trainer.train(num_epochs)
    
    
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    
    
    # Plot loss for first model (task 2c)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history_new["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history_new["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.show
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, .99])
    utils.plot_loss(train_history_new["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history_new["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task2c_train_loss.png")
    plt.show()