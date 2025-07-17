# A7-A- R
# Load Libraries
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)
library(imputeTS)
library(randomForest)
library(rpart)
library(keras)
library(tsibble)
library(feasts)

# Load the CSV
df <- read.csv("C:\\Users\\hp\\Downloads\\Taiwan Semiconductor Stock Price History.csv")
head(df)

# Convert Date to Date type and arrange by date
df$Date <- mdy(df$Date)
df <- df[order(df$Date), ]
head(df)

# Clean 'Change %' and 'Vol.' columns (optional if not needed)
# Rename the columns for ease of use
colnames(df)[colnames(df) == "Change.."] <- "ChangePercent"
colnames(df)[colnames(df) == "Vol."] <- "Volume"

# Clean Change % and Volume columns
# Rename the columns for ease of use
colnames(df)[colnames(df) == "Change.."] <- "ChangePercent"
colnames(df)[colnames(df) == "Vol."] <- "Volume"

# Clean Change % and Volume columns
df$ChangePercent <- as.numeric(gsub("%", "", df$ChangePercent))
df$Volume <- as.numeric(gsub("M", "e6", gsub("K", "e3", df$Volume)))# Converts volume to numeric

# Interpolate missing values (if any)
df$Price <- na_interpolation(df$Price)

# Plot line chart
ggplot(df, aes(x = Date, y = Price)) +
  geom_line(color = "steelblue") +
  labs(title = "TSMC Daily Closing Price", x = "Date", y = "Price")

# Train-test split (80-20)
train_size <- floor(0.8 * nrow(df))
train <- df[1:train_size, ]
test <- df[(train_size + 1):nrow(df), ]

# Convert to monthly and take average
monthly_ts <- df %>%
  select(Date, Price) %>%
  tsibble(index = Date) %>%
  index_by(month = ~ yearmonth(.)) %>%
  summarise(avg_price = mean(Price)) %>%
  as_tsibble()

monthly_ts_ts <- ts(monthly_ts$avg_price, frequency = 12)

# Decomposition
decomp_add <- decompose(monthly_ts_ts, type = "additive")
decomp_mul <- decompose(monthly_ts_ts, type = "multiplicative")
# Clear graphics parameters
dev.off()        # Closes any open graphics device
graphics.off()   # Closes all devices (use only if needed)

# Plot decomposition correctly
plot(decomp_add)
title("Additive Decomposition")

plot(decomp_mul)
title("Multiplicative Decomposition")



# --------------------------------------
# Holt-Winters Forecast (Monthly)
# --------------------------------------
hw_model <- HoltWinters(monthly_ts_ts)
hw_forecast <- forecast(hw_model, h = 12)
plot(hw_forecast, main = "Holt-Winters Forecast - 12 months")

# Plot the forecast first
plot(hw_forecast, main = "Holt-Winters Forecast (12 months)", col = "blue")

# Add fitted values over the original time series
lines(hw_model$fitted[,1], col = "darkgreen")  # [,1] extracts the fitted level

# Plot actual series + forecast
ts.plot(hw_model$x, hw_forecast$mean, col = c("black", "blue"), lty = 1:2)
legend("topleft", legend = c("Actual", "Forecast"), col = c("black", "blue"), lty = 1:2)




# --------------------------------------
# ARIMA Daily
# --------------------------------------
daily_ts <- ts(df$Price, frequency = 365)
arima_model <- auto.arima(daily_ts)
checkresiduals(arima_model)
forecast_arima <- forecast(arima_model, h = 90)
plot(forecast_arima, main = "ARIMA Forecast - Daily (3 months)")

plot(forecast_arima, main = "ARIMA Daily Forecast (Next 90 Days)", col = "purple")
lines(fitted(arima_model), col = "gray")
legend("topleft", legend = c("Forecast", "Fitted"), col = c("purple", "gray"), lty = 1)


# SARIMA
sarima_model <- auto.arima(daily_ts, seasonal = TRUE)
forecast_sarima <- forecast(sarima_model, h = 90)
plot(forecast_sarima, main = "SARIMA Forecast - 3 Months")

plot(forecast_sarima, main = "SARIMA Daily Forecast (Next 90 Days)", col = "red")
lines(fitted(sarima_model), col = "darkorange")
legend("topleft", legend = c("Forecast", "Fitted"), col = c("red", "darkorange"), lty = 1)


# ARIMA on Monthly
arima_monthly <- auto.arima(monthly_ts_ts)
forecast_monthly <- forecast(arima_monthly, h = 12)
plot(forecast_monthly, main = "ARIMA Forecast - Monthly")

plot(forecast_monthly, main = "ARIMA Monthly Forecast (Next 12 Months)", col = "brown")
lines(fitted(arima_monthly), col = "darkcyan")
legend("topleft", legend = c("Forecast", "Fitted"), col = c("brown", "darkcyan"), lty = 1)


# --------------------------------------
# MACHINE LEARNING: Data Prep
# --------------------------------------
df_ml <- df %>%
  mutate(Lag1 = lag(Price, 1),
         Lag2 = lag(Price, 2)) %>%
  drop_na()

train_ml <- df_ml[1:train_size, ]
test_ml <- df_ml[(train_size + 1):nrow(df_ml), ]

# Decision Tree
tree_model <- rpart(Price ~ Lag1 + Lag2, data = train_ml)
tree_pred <- predict(tree_model, test_ml)
tree_rmse <- sqrt(mean((test_ml$Price - tree_pred)^2))

# Random Forest
rf_model <- randomForest(Price ~ Lag1 + Lag2, data = train_ml)
rf_pred <- predict(rf_model, test_ml)
rf_rmse <- sqrt(mean((test_ml$Price - rf_pred)^2))

print(paste("Decision Tree RMSE:", round(tree_rmse, 2)))
print(paste("Random Forest RMSE:", round(rf_rmse, 2)))

library(ggplot2)
test_ml$DT_Pred <- tree_pred

ggplot(test_ml, aes(x = Date)) +
  geom_line(aes(y = Price, color = "Actual")) +
  geom_line(aes(y = DT_Pred, color = "Decision Tree Prediction")) +
  labs(title = "Decision Tree Prediction vs Actual",
       x = "Date", y = "Price") +
  scale_color_manual(values = c("Actual" = "black", "Decision Tree Prediction" = "blue"))

test_ml$RF_Pred <- rf_pred

ggplot(test_ml, aes(x = Date)) +
  geom_line(aes(y = Price, color = "Actual")) +
  geom_line(aes(y = RF_Pred, color = "Random Forest Prediction")) +
  labs(title = "Random Forest Prediction vs Actual",
       x = "Date", y = "Price") +
  scale_color_manual(values = c("Actual" = "black", "Random Forest Prediction" = "darkgreen"))



# --------------------------------------
# LSTM Neural Network (optional - requires Keras setup)
# --------------------------------------
library(keras)
install_tensorflow()
install_tensorflow(method = "virtualenv")
library(tensorflow)
tf$constant("Hello from TensorFlow")

# Normalization
price_series <- df_ml$Price
price_scaled <- scale(price_series)

# Create sequences for LSTM
lag <- 3
X <- embed(price_scaled, lag)[, -1]
y <- embed(price_scaled, lag)[, 1]

# Train-Test for LSTM
X_train <- X[1:train_size, ]
X_test <- X[(train_size+1):nrow(X), ]
y_train <- y[1:train_size]
y_test <- y[(train_size+1):length(y)]

# Reshape for Keras [samples, timesteps, features]
X_train_reshape <- array(X_train, dim = c(nrow(X_train), lag - 1, 1))
X_test_reshape <- array(X_test, dim = c(nrow(X_test), lag - 1, 1))

# LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(lag - 1, 1)) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = 'mse',
  optimizer = 'adam'
)

model %>% fit(X_train_reshape, y_train, epochs = 20, verbose = 1)

pred_lstm <- model %>% predict(X_test_reshape)
lstm_rmse <- sqrt(mean((y_test - pred_lstm)^2))
print(paste("LSTM RMSE:", round(lstm_rmse, 2)))


# Optional: Only run if keras part is trained
if (exists("pred_lstm")) {
  # Rescale predicted and actual if needed
  lstm_pred_plot <- tibble(
    Date = df_ml$Date[(train_size + 3):nrow(df_ml)],  # Adjust index if lag=3
    Actual = y_test,
    Predicted = as.vector(pred_lstm)
  )
  
  ggplot(lstm_pred_plot, aes(x = Date)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = Predicted, color = "LSTM Prediction")) +
    labs(title = "LSTM Forecast vs Actual", x = "Date", y = "Price") +
    scale_color_manual(values = c("Actual" = "black", "LSTM Prediction" = "red"))
}
