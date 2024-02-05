# Improved-Total-Variation-Denoising-Model

## Group I. Image Denoising Problem I

The goal of this case study is to construct a smoothing term that reduces noise without a staircase effect.

- Consider the following energy function:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\min_u\{\frac{1}{2}\|f-u\|_2^2+\lambda\|\nabla{u}\|_1\}),

  where ![equation](https://latex.codecogs.com/svg.latex?\color{white}f=u+n) , f is the original image, n is the Gaussian noise, ![equation](https://latex.codecogs.com/svg.latex?\color{white}\|\cdot\|_2) and ![equation](https://latex.codecogs.com/svg.latex?\color{white}\|\cdot\|_1) are ![equation](https://latex.codecogs.com/svg.latex?\color{white}L_2) and ![equation](https://latex.codecogs.com/svg.latex?\color{white}L_1) norms, respectively. As we know, the model \(0.1\) is the famous image denoising model called TV model.

  Please illustrate what are the major drawbacks of the model \(0.1\) and show your numerical results to demonstrate your point.
- Now consider a modified version of the model \(0.1\) as follows:

  ![equation](https://latex.codecogs.com/svg.latex?\color{white}\min_u\{\frac{1}{2}\|f-u\|_2^2+\lambda\|g\nabla{u}\|_1\}),

  where ![equation](https://latex.codecogs.com/svg.latex?\color{white}g%20=%20\frac{1}{1+|\nabla%20G%20*%20f|^2}) , ![equation](https://latex.codecogs.com/svg.latex?\color{white}G_\sigma%20\ast%20f) denotes convolving image f with a Gaussian kernel whose standard deviation is ![equation](https://latex.codecogs.com/svg.latex?\color{white}\sigma). Please indicate what's the function g role in smoothing the image and show your numerical results for image denoising using the model \(0.2\).
- Can you design a regularization term or make some adjustment for the model \(0.1\)? Please explain what's the reason for your choice?
