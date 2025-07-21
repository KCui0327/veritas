=================================================================
    Layer (type)           Output Shape         Parameters     Trainable
=================================================================
    Embedding            [1,   100,   128]      2,560,000    Yes
    LSTM                 [1,   100,   256]        263,168    Yes
    Dropout              [1,   256]                 0       No
    Linear(fc1)          [1,    64]              16,448     Yes
    Linear(fc2)          [1,     1]                 65     Yes
    Sigmoid              [1]                        0       No
=================================================================
Total params: 2,839,681
Trainable params: 2,839,681
Non-trainable params: 0
