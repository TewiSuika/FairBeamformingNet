# FairBeamformingNet
Beamforming of Transmit Antennas Using Fair Hybrid Beamforming Network for Performance Enhancement of Integrated Sensing And Communications

# main.py - Main Program Module
Functional Description:
1.Serves as the system entry point, coordinating the workflow of the entire beamforming system.
2. Loads or trains deep learning models.
3. Compares the performance of different beamforming algorithms.
4. Visualizes results (beam patterns, rates, BER, CRLB, etc.).

# Models.FBN_model.py - Deep Learning Model Module
Functional Description:
1.Defines the ISAC beamforming neural network.
2. Multi-task loss function (MultiTaskLoss).
3. Model management tools (save/load).
4. Hybrid beamforming implementation (analog + digital).



# Models.channel_data_generate.py-Channel Generation Module
Functional Description:
1.Generates communication and sensing channel data.
2.Creates antenna array response vectors.
3.Generates multipath communication channels (LOS + NLOS).
4.Creates training datasets.


# Models.training.py - Model Training Module
Functional Description:
1.Data generation and preprocessing.
2.Dataset splitting (training set/validation set).
3.Model training loop.
4.Validation and model saving.


# Algorithms.optimization.py - Optimization Algorithm Module
Functional Description:
1.Hybrid beamforming objective function.
2.Implementation of various optimization algorithms (DE/PSO/GWO/WOA).
3.Interface with traditional methods (ZF/MMSE).
4.Weight conversion tools.

# evaluation.py - Performance Evaluation Module
Functional Description:
1.Beam pattern analysis and visualization.
2. User communication rate calculation.
3. Target sensing performance evaluation (CRLB).
4. Bit error rate (BER) calculation.

# Configuration File config.py
Global system parameter configuration:
1.Hardware parameters: number of antennas, wavelength, etc.
2.Scenario parameters: user angles, target angles.
3.Algorithm parameters: rho trade-off factor, SNR range.
4.Training parameters: hidden layer size, batch size, etc.


