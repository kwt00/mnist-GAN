# MNIST GAN Project
MNIST GAN is a generative adversarial network designed to teach itself to be able to mimick handwritten digits 0-9, training on data from the MNIST(Modified National Institute of Standards and Technology database) dataset. 

In this model, I used the amazing Tensorflow Keras library, recommended nerual network design elements, and much trial and error to find a model configuration with high levels of accuracy.

***A complete and detailed series of explanations is included via comments in the Python code, so if you are wondering how exactly the model works and what each component does, feel free to check out the program file itself.***

In its current state, here is an example of some test output:

![gen3](https://user-images.githubusercontent.com/116334641/207789586-4341bab9-7582-4b03-be78-0a1df82f2938.png)
![gen4](https://user-images.githubusercontent.com/116334641/207789598-07912438-2851-4f03-b317-220bee483f10.png)
![gen8](https://user-images.githubusercontent.com/116334641/207789612-b6f84f65-ceaa-41b7-b539-219e09940ae4.png)

While the testing phase only included 10 epochs, the model exhibited decently high levels of accuracy and produced moderately legible digits. If trained more thourougly, perhaps 40-80 epochs, the model could be greatly improved upon. 

To try out the program, simply download the file and paste it into your favorite python notebook editor(these usually provide easier ways to install the libraries than normal IDEs).

The model does not come pre-trained, and training times for a single epoch can vary from 20-30 minutes.
