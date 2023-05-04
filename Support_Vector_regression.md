# Support Vector Regression

## The difference between Support Vector Regression (SVR) and normal regression, and the intuition behind using different kernel functions.

Normal regression (e.g., linear regression) aims to find a function that minimizes the sum of the differences between the predicted values and the actual values. The primary focus is on minimizing the error in the predictions, which can make it sensitive to outliers and noise in the data. In linear regression, we fit a straight line to the data in 2D space.

SVR, on the other hand, doesn’t focus on minimizing the error. Instead, it tries to fit a function within an epsilon-insensitive tube around the actual values (ignoring the points inside the tube). The focus is on maximizing the margin between the function and the nearest data points outside the tube (support vectors). This approach makes SVR less sensitive to outliers and noise, and it can produce more robust models.

Now, let’s discuss the intuition behind using different kernel functions in SVR. In simple terms, kernel functions enable SVR to learn more complex, non-linear relationships between the predictor variables and the target variable.

While using a linear kernel results in a straight line (as shown in the previous example), using other kernel functions can generate different shapes in 2D space that capture more complex relationships in the data. For instance, a radial basis function (RBF) kernel can produce smooth curves or even a “bell-shaped” function, while a polynomial kernel can produce curved functions of varying degrees.

Visualizing the intuition behind non-linear kernel functions in SVR can be done by transforming a 2D scatter plot of the data (with x on the horizontal axis and y on the vertical axis) as follows:

1. Using a linear kernel, imagine fitting a straight line that lies within an epsilon-insensitive tube, while maximizing the margin between the line and the nearest data points outside the tube.

2. Using an RBF kernel, imagine fitting a smooth curve, and in some cases forming a “bell-shaped” function, that lies within the epsilon-insensitive tube, while maximizing the margin between the curve and the nearest data points outside the tube.

3. Using a polynomial kernel, imagine fitting a curved function of varying degrees (e.g., quadratic, cubic) that lies within the epsilon-insensitive tube, while maximizing the margin between the curve and the nearest data points outside the tube.

Using different kernel functions allows SVR to capture non-linear relationships between the predictor and target variables, and to create more flexible regression models. This adaptability to various types of data is what sets SVR apart from normal linear regression.


Consider a 2D scatter plot of the ten data points with x (feature) on the horizontal axis and y (target) on the vertical axis.

Because the RBF kernel is a non-linear kernel, it can handle complex relationships between the predictor and target variables. Essentially, it maps the data points to a higher-dimensional space where a linear separation becomes possible. However, the result of this transformation in the higher-dimensional space is a smooth curve when viewed in the original 2D space.

Now, let’s imagine fitting an RBF kernel-based SVR model to the data:

1. The RBF kernel takes a parameter called gamma, which controls the level of flexibility or smoothness in the fitted curve. A smaller gamma value results in a smoother curve while a larger gamma value creates a tighter (sharper) curve around the data points.

2. The SVR model tries to find a function that lies within an epsilon-insensitive tube, which we can visualize as a flexible “band” around the actual data points. The tube has a width defined by the epsilon value.

3. When using the RBF kernel, the SVR algorithm will produce a curved function that lies within this epsilon-insensitive tube, while maximizing the margin between the curve and the nearest data points outside the tube (support vectors).

4. In some cases, this curve can have multiple “peaks” or “valleys” as it tries to stay within the tube and capture more complex non-linear relationships in the data.

5. Because the RBF kernel can create smooth, “bell-shaped” curves, the SVR model may not pass directly through all data points. Instead, it will strike a balance between capturing the overall data pattern and staying within the epsilon-insensitive tube.
