import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM, Dense, Dropout, Input

def create_cnn_lstm_model(input_shape):
    model = Sequential()

    # Khối CNN 2D
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    # Khối Flatten và LSTM
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(64))
    model.add(Dropout(0.25))

    # Đầu ra
    model.add(Dense(20, activation='softmax'))

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

input_shape = (5, 64, 64, 3)  # Depth (frames), Height, Width, Channels

model = create_cnn_lstm_model(input_shape)
model.summary()

# Load trọng số đã lưu
model.load_weights('CNNLSTM_model.weights.h5')
