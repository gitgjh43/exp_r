library(glmnet)

# n: number of rows
# p: number of features
# b0: beta_0 [intercept]
# b: beta vector [b1, b2, ..., bp]

run_bias_variance_experiment <- function() {	
	# number of observations in the test data set [to approximate the *true* test error, we use a very large test data set]
	n.test = 10 * 1000
	# number of observations in each training data set
	n.train = 50
	# "number_of_training_datasets" number of training data sets will be used to estimate the expected values of (1) test MSE, (2) bias^2, and (3) variance
	number_of_training_datasets = 200

	# parameters for the true model
	b0 = 0.5
	b = runif(45, min=-1, max=1)
	p = length(b)
	noise.sd = 2

	# use this vector of coefficients (b) for the experiment with 2 non-zero coefficents and 43 zero coefficients	
	#b = c(6.1, -4.2, rep(0, 43))
	
	# the lambdas for glmnet()
	labmda.count = 100
	# Supply a *decreasing* sequence of ‘lambda’ values
	lambda.grid = 10 ^ seq(5, -4, length=labmda.count)

	# sample the test data set
	X.test = sample_X(n.test, p)	
	r = sample_y(b0, b, X.test, noise.sd)
	y.test = r$Y
	y.test.without_noise = r$Y.without_noise
	
	# sanity check: too much noise?
	stopifnot( var(y.test) > 3*(noise.sd ^ 2) )
	
	# sample "number_of_training_datasets" number of training data sets
	X.train.list = vector(mode='list', length=number_of_training_datasets)
	y.train.list = vector(mode='list', length=number_of_training_datasets)
	for (i in 1:number_of_training_datasets) {
		X.train.list[[i]] = sample_X(n.train, p)	
		y.train.list[[i]] = sample_y(b0, b, X.train.list[[i]], noise.sd)$Y	
	}
	
	# for each alpha:
	#		(1) for each training data set, train(), and then predict() for the test data set
	#		(2) compute bias, variance, and MSE
	#		(3) plot bias, variance, and MSE
	par(mfrow=c(1, 3))
	results = vector(mode='list', length=2)
	for (alpha in c(0, 1)) {
		# (1) and (2) 
		r = train_predict__and__compute_bias_variance_MSE(X.train.list, y.train.list, lambda.grid, X.test, y.test, y.test.without_noise, noise.sd, alpha, n.test, labmda.count, number_of_training_datasets)
		results[[alpha + 1]] = r
		# (3)
		plot_bias_variance_curve(lambda.grid, r$MSE.vector, r$bias_squared.vector, r$variance.vector, noise.sd, title=sprintf('alpha = %d', alpha))	
	}
	
	plot_bias_variance_curve.both(log(results[[1]]$MSE.train.vector), results[[1]]$MSE.vector, results[[1]]$bias_squared.vector, results[[1]]$variance.vector, 
								log(results[[2]]$MSE.train.vector), results[[2]]$MSE.vector, results[[2]]$bias_squared.vector, results[[2]]$variance.vector, 
								noise.sd)
}

# plot both ridge and lasso
# X axis: log(training MSE)
plot_bias_variance_curve.both <- function(MSE.train.vector.ridge, MSE.vector.ridge, bias_squared.vector.ridge, variance.vector.ridge, 
										MSE.train.vector.lasso, MSE.vector.lasso, bias_squared.vector.lasso, variance.vector.lasso, 
										noise.sd) {
	xlim = range(c(MSE.train.vector.ridge, MSE.train.vector.lasso))
	ylim = range(c(MSE.vector.ridge, bias_squared.vector.ridge, variance.vector.ridge, MSE.vector.lasso, bias_squared.vector.lasso, variance.vector.lasso))
	plot(0, 0, xlab='log(Training MSE)', ylab='MSE', main='Ridge vs Lasso', xlim=xlim, ylim=ylim, type='n')
	
	matlines(MSE.train.vector.ridge, cbind(MSE.vector.ridge, bias_squared.vector.ridge, variance.vector.ridge), col=c('blue', 'green', 'red'), type='l', lwd=1.5, lty=1)
	matlines(MSE.train.vector.lasso, cbind(MSE.vector.lasso, bias_squared.vector.lasso, variance.vector.lasso), col=c('blue', 'green', 'red'), type='l', lwd=1.5, lty=2)
	
	abline(h=(noise.sd ^ 2), col='black')
	legend('topright', legend=c('MSE [Ridge]', 'Bias ^ 2 [Ridge]', 'Variance [Ridge]', 'MSE [Lasso]', 'Bias ^ 2 [Lasso]', 'Variance [Lasso]', 'Var(noise)'), 
			col=c('blue', 'green', 'red', 'blue', 'green', 'red', 'black'), lty=c(1, 1, 1, 2, 2, 2, 1))
}

plot_bias_variance_curve <- function(lambda.grid, MSE.vector, bias_squared.vector, variance.vector, noise.sd, title) {
	matplot(log10(lambda.grid), cbind(MSE.vector, bias_squared.vector, variance.vector), col=c('blue', 'green', 'red'), type='l', lwd=1.5, lty=1, xlab='log(lambda)', ylab='MSE', main=title)
	abline(h=(noise.sd ^ 2), col='black')
	legend('topright', legend=c('MSE', 'Bias ^ 2', 'Variance', 'Var(noise)'), col=c('blue', 'green', 'red', 'black'), lty=1)
}

train_predict__and__compute_bias_variance_MSE <- function(X.train.list, y.train.list, lambda.grid, X.test, y.test, y.test.without_noise, noise.sd, alpha, n.test, labmda.count, number_of_training_datasets) {
	# "pred.3darray" array contains "number_of_training_datasets" number of 2D arrays (matrices)
	#		each 2D array (matrix) contains test set predictions [output of train_and_predict()] [dim = (n.test, labmda.count)]
	pred.3darray = array(NA, dim=c(n.test, labmda.count, number_of_training_datasets))
	
	# one row for each training data set
	# one col for each lambda
	MSE.train = matrix(NA, nrow=number_of_training_datasets, ncol=labmda.count)
	
	for (i in 1:number_of_training_datasets) {
		r = train_and_predict(X.train.list[[i]], y.train.list[[i]], lambda.grid, X.test, alpha)
		pred.3darray[, , i] = r$pred
		MSE.train[i, ] = r$MSE.train.vector
	}
	
	r = compute_bias_variance_MSE(pred.3darray, labmda.count, y.test, y.test.without_noise)
	# each of the following three vectors has "labmda.count" number of elements (one for each value of λ)
	MSE.vector          = r$MSE.vector
	bias_squared.vector = r$bias_squared.vector
	variance.vector     = r$variance.vector
	
	# sanity test: MSE decomposes into (1) variance, (2) bias ^ 2, and (2) var(noise)
	MSE.vector.1 = variance.vector + bias_squared.vector + (noise.sd ^ 2)
	stopifnot(max(100 * abs(MSE.vector.1 - MSE.vector) / abs(MSE.vector))   <   2) # we tolerate up to 2% deviation
	
	# compute training error
	MSE.train.vector = apply(MSE.train, 2, mean)
	
	return(list(MSE.vector=MSE.vector, bias_squared.vector=bias_squared.vector, variance.vector=variance.vector, MSE.train.vector=MSE.train.vector))
}

compute_bias_variance_MSE <- function(pred.3darray, labmda.count, y.test, y.test.without_noise) {
	stopifnot(dim(pred.3darray)[2] == labmda.count)
	
	MSE.vector          = rep(NA, labmda.count)
	bias_squared.vector = rep(NA, labmda.count)
	variance.vector     = rep(NA, labmda.count)
	for (lamda_index in 1:labmda.count) {
		# a: a matrix of predictions for a given lambda
		# 	number of rows = n.test (number of test rows)
		# 	number of cols = number_of_training_datasets (one column per training data set)
		a = pred.3darray[, lamda_index, ]
		
		# apply(a, 1, var): variance (of predictions) for each test row [each test row has "number_of_training_datasets" number of predictions]
		#	mean( apply(...) ): expected value of the variance
		variance.vector[lamda_index] = mean( apply(a, 1, var) )
		
		# avg: average (of predictions) for each test row [each test row has "number_of_training_datasets" number of predictions]
		avg = apply(a, 1, mean) 
		bias_squared = mean(   (avg - y.test.without_noise) ^ 2   )
		bias_squared.vector[lamda_index] = bias_squared
		
		delta = a - y.test
		MSE.vector[lamda_index] = mean(delta ^ 2)
	}
	
	list(MSE.vector=MSE.vector, bias_squared.vector=bias_squared.vector, variance.vector=variance.vector)
}

train_and_predict <- function(X.train, y.train, lambda.grid, X.test, alpha) {
	glmnet.fit = glmnet(X.train, y.train, family='gaussian', alpha=alpha, lambda=lambda.grid, thresh=1e-07) 
	
	# predict() returns a matrix
	#	number of rows = nrow(newx)
	#	number of cols = number of lambdas to try (one for each value of λ)
	pred = predict(glmnet.fit, newx=X.test)
	
	# compute MSE.train.vector
	pred.train = predict(glmnet.fit, newx=X.train)
	delta = pred.train - y.train
	MSE.train.vector = apply(delta ^ 2, 2, mean)	# MSE for each lambda; length = number of lambdas to try (one for each value of λ)
	
	list(pred=pred, MSE.train.vector=MSE.train.vector)
}

sample_X <- function(n, p ) {
	matrix(rnorm(n*p), nrow=n, ncol=p)
}

sample_y <- function(b0, b, X, noise.sd) {
	stopifnot(ncol(X) == length(b))
	
	X1 = cbind(1, X)
	b1 = c(b0, b)
	Y.without_noise = X1 %*% b1
	Y.without_noise = c(Y.without_noise) # convert from nx1 matrix to a column vector
	
	noise = rnorm(nrow(X), mean = 0, sd = noise.sd)
	
	Y = Y.without_noise + noise
	
	list(Y=Y, Y.without_noise=Y.without_noise)
}
