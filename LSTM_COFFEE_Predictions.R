
##load libraries
library(keras)
library(tensorflow)
library(ggplot2)
library(stats)
library(readr)
library(dplyr)
library(forecast)
library(Metrics)
library(scales)

##load two outright contracts let's try coffee
Contract1 <- 'KC.20191218'
Contract2 <- 'KC.20190918'
EndDate   <- '2019-07-31'
StartDate <- '2019-05-30'
DaysBack  <- 30

KCZ19 <- Pull.AWS.Data(Contract1, '5min', StartDate, EndDate, "close")
KCU19 <- Pull.AWS.Data(Contract2, '5min', StartDate, EndDate, "close")

Price.Data            <- cbind(KCZ19, KCU19)
colnames(Price.Data)  <- c('KCZ19.WMP', 'KCU19.WMP')
##transform to stationary data
Price.Data$CS.WMP     <- Price.Data$KCZ19.WMP - Price.Data$KCU19.WMP

##explore the data
plot(Price.Data$KCZ19.WMP)
plot(Price.Data$KCU19.WMP)
plot(Price.Data$CS.WMP)

##Create a lagged series for point predictions at k-steps & insert zeros for unknown
Price.Data$CS.WMP.1hrlag <- lag(Price.Data$CS.WMP, 12)
Price.Data$CS.WMP.1hrlag[is.na(Price.Data$CS.WMP.1hrlag)] <- 0

##Apply Normalization to the Fields. The activation funciton for neural nets is the sigmoid function which take on values between -1 and 1
##min-max normalization
Price.Data$CS.WMP.rescaled <- rescale(as.numeric(Price.Data$CS.WMP), to = c(-1, 1)) 
Price.Data$CS.WMP.1hrlag.rescaled <- rescale(as.numeric(Price.Data$CS.WMP.1hrlag), to = c(-1, 1)) 

##we will try to predict the future 1 hour value from the current value
##x_train <- Price.Data$CS.WMP.1hrlag.rescaled
##y_train <- Price.Data$CS.WMP.rescaled

input_matrix <- Price.Data$CS.WMP.rescaled
##need to reformulate the data in batches so it can be processed by the LSTM
n_timesteps <- 12    ## 5 minute increments at 12 will give hourly predictions
n_predictions <- n_timesteps
batch_size <- 10

# functions used to create batches to be feed into network found on rstudio blog
build_matrix <- function(tseries, overall_timesteps) {
  t(sapply(1:(length(tseries) - overall_timesteps + 1), function(x) 
    tseries[x:(x + overall_timesteps - 1)]))
}

reshape_X_3d <- function(X) {
  dim(X) <- c(dim(X)[1], dim(X)[2], 1)
  X
}

train_matrix <- build_matrix(as.matrix(input_matrix), n_timesteps + n_predictions)
##build the batches of 12 predictions that you want to predict
X_train <- train_matrix[, 1:n_timesteps]
y_train <- train_matrix[, (n_timesteps + 1):(n_timesteps * 2)]
X_train <- X_train[1:(nrow(X_train) %/% batch_size * batch_size), ]
y_train <- y_train[1:(nrow(y_train) %/% batch_size * batch_size), ]

##add third axis and force 3 dimensions
y_train_3d <- reshape_X_3d(y_train)
x_train_3d <- reshape_X_3d(X_train)

##source config file with all the LSTM setting and parameters
source('~/LSTM Predictions/LSTM_config.R')

###constructing and training the model -- Short Version

# create the model
model <- keras_model_sequential()

# add layers
# we have just two, the LSTM and the time_distributed 
model %>%
  layer_lstm(
    units = FLAGS$n_units, 
    # the first layer in a model needs to know the shape of the input data
    batch_input_shape  = c(FLAGS$batch_size, FLAGS$n_timesteps, n_features),
    dropout = FLAGS$dropout,
    recurrent_dropout = FLAGS$recurrent_dropout,
    # by default, an LSTM just returns the final state
    return_sequences = TRUE
  ) %>% time_distributed(layer_dense(units = 1))

model %>%
  compile(
    loss = FLAGS$loss,
    optimizer = optimizer,
    # in addition to the loss, Keras will inform us about current 
    # MSE while training
    metrics = list("mean_squared_error")
  )

history <- model %>% fit(
  x          = x_train_3d,
  y          = y_train_3d,
  batch_size = FLAGS$batch_size,
  epochs     = FLAGS$n_epochs,
  callbacks = callbacks
)

## score the dataset
pred_train <- model %>%
  predict(x_train_3d, batch_size = FLAGS$batch_size) %>%
  .[, , 1]

# Retransform values to original scale
# pred_train <- (pred_train * scale_history + center_history) ^2
compare_train <- input_matrix

# build a dataframe that has both actual and predicted values
for (i in 1:nrow(pred_train)
) {
  varname <- paste0("pred_train", i)
  
  compare_train <-
    mutate(as.data.frame(compare_train),
           !!varname := c(rep(NA,
                              FLAGS$n_timesteps + i - 1
           ),
          pred_train[i,],
           rep(NA, 
               nrow(compare_train) - FLAGS$n_timesteps * 2 - i + 1
           )
           )
    )
}

##We compute the average RSME over all sequences of predictions.

coln <- colnames(compare_train)[4:ncol(compare_train)]

cols <- coln %>% 
  map(~ sym(.x)
  )

# RMSE
rmse_test <-
  map(cols,
      function(col)
        rmse(data = compare_test,
             truth = CS.WMP.rescaled,
             estimate = !!col,
             na.rm = TRUE
        ) %>% 
        select(.estimate)
  ) %>% 
  bind_rows()

rmse_test <- mean(rmse_test$.estimate, na.rm = TRUE)

rmse_test
21.01495


##How do these predictions really look? As a visualization of all predicted sequences would look pretty crowded, we arbitrarily pick start points at regular intervals.
compare_train$index <- index(input_matrix)
sampleset <- compare_train[1:100,]

##example of batch prediction plot
ggplot(sampleset, aes(x = index, y = CS.WMP.rescaled)) + geom_line() +
  geom_line(aes(y = pred_train1), color = "cyan") +
  geom_line(aes(y = pred_train13), color = "red") +
  geom_line(aes(y = pred_train25), color = "green") +
  geom_line(aes(y = pred_train38), color = "violet") +
  geom_line(aes(y = pred_train50), color = "cyan") +
  geom_line(aes(y = pred_train62), color = "red") 



