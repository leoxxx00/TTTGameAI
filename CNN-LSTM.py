import tkinter as tk
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import os
import csv

# CSV file to store game data
CSV_FILE = "tic_tac_toe_data.csv"

# TicTacToe game logic
class TicTacToe:
    def __init__(self):
        self.board = [None] * 9
        self.current_player = random.choice(['X', 'O'])  # Randomly select starting player
        self.move_sequence = []

    def make_move(self, position):
        if self.board[position] is None:
            self.board[position] = self.current_player
            self.move_sequence.append(self.board[:])  # Log board state for CNN+LSTM learning
            print(f"Move made: {self.current_player} at position {position}")
            print(f"Current board state: {self.board}")
            return True
        return False

    def check_winner(self):
        # Winning combinations
        winning_combos = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]

        for combo in winning_combos:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] and self.board[combo[0]] is not None:
                return self.board[combo[0]]

        if all(cell is not None for cell in self.board):
            return 'Draw'

        return None

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        print(f"Switched player to {self.current_player}")

    def reset(self):
        self.board = [None] * 9
        self.move_sequence = []  # Reset move sequence
        self.current_player = random.choice(['X', 'O'])  # Randomly select who starts
        print(f"Game board reset. {self.current_player} starts!")


# AI agent for the game
class CNNLSTMAgent:
    def __init__(self):
        self.model = self.build_model()
        self.encoder = OneHotEncoder(sparse_output=False)  # Fit encoder to the possible board states
        self.encoder.fit(np.array([['X'], ['O'], [None]]))
        self.kmeans = None  # Placeholder for the KMeans clustering model

        # Increase randomness
        self.epsilon = 1.0  # Initial exploration rate, keep high for more randomness
        self.epsilon_min = 0.3  # Higher minimum exploration rate for ongoing randomness
        self.epsilon_decay = 0.99  # Slower decay rate to maintain randomness for longer

        self.load_csv_data()

    def build_model(self):
        model = Sequential()

        # TimeDistributed CNN for processing each board state
        model.add(TimeDistributed(Conv2D(32, (2, 2), activation='relu'), input_shape=(None, 3, 3, 1)))
        model.add(TimeDistributed(Flatten()))

        # LSTM for processing the sequence of board states
        model.add(LSTM(128, return_sequences=False))

        # Fully connected layer for choosing actions
        model.add(Dense(9, activation='softmax'))
        model.compile(optimizer=Adam(), loss='mse')
        print("Model built")
        return model

    def preprocess_board(self, board):
        # Convert the board to a 3x3 matrix and one-hot encode it
        if len(board) != 9:
            raise ValueError(f"Invalid board length: {len(board)}. Expected 9 elements.")

        board_reshaped = np.array(board).reshape(3, 3)
        board_encoded = self.encoder.transform(board_reshaped.reshape(-1, 1)).reshape(3, 3, 3)
        return board_encoded[:, :, 0:1]  # Use first dimension for one-hot encoding

    def fit_kmeans(self, board_states, n_clusters=5):
        # Flatten the board states to a 1D vector for clustering
        flattened_states = [board.reshape(-1) for board in board_states]
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(flattened_states)
        print(f"K-Means clustering fitted with {n_clusters} clusters.")

    def get_cluster_label(self, board_state):
        # Flatten the board state and get the cluster label
        if self.kmeans:
            flattened_state = board_state.reshape(-1).reshape(1, -1)
            cluster_label = self.kmeans.predict(flattened_state)
            return cluster_label[0]
        return None

    def choose_action(self, board, available_actions):
        # Epsilon-greedy strategy for exploration and exploitation
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        else:
            # Preprocess the current board
            board_encoded = self.preprocess_board(board)
            board_encoded = np.expand_dims(board_encoded, axis=0)  # Add batch dimension
            board_encoded = np.expand_dims(board_encoded, axis=0)  # Add time step dimension

            # Predict the best move using CNN + LSTM
            predictions = self.model.predict(board_encoded)[0]

            # Choose the best available move
            available_predictions = [(i, predictions[i]) for i in available_actions]
            move = max(available_predictions, key=lambda x: x[1])[0]
            print(f"Predicted move probabilities: {predictions}")
            return move

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_model(self, move_sequence, reward):
        X = []
        y = []
        for i in range(len(move_sequence)):
            board_encoded = self.preprocess_board(move_sequence[i])
            X.append(board_encoded)

            cluster_label = self.get_cluster_label(board_encoded)
            if cluster_label is not None:
                cluster_one_hot = np.eye(5)[cluster_label].flatten()
                board_encoded_with_cluster = np.concatenate([board_encoded.flatten(), cluster_one_hot], axis=0)
            else:
                board_encoded_with_cluster = board_encoded.flatten()

            move_targets = np.zeros(9)
            move_targets[np.argmax(board_encoded.reshape(-1))] = reward

            y.append(move_targets)

        X = np.array(X)
        y = np.array(y)

        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)

        print("Training model on new data with cluster features...")
        self.model.fit(X, y, epochs=300, verbose=1)  # Increased epochs to 300

    def save_model(self):
        self.model.save("tic_tac_toe_cnn_lstm.h5")
        print("Model saved")

    def load_model(self):
        if os.path.exists("tic_tac_toe_cnn_lstm.h5"):
            self.model = tf.keras.models.load_model("tic_tac_toe_cnn_lstm.h5", compile=False)
            self.model.compile(optimizer=Adam(), loss='mse')
            print("Model loaded and recompiled with mse loss.")

    def append_to_csv(self, move_sequence, reward):
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            for move in move_sequence:
                move_flat = [str(x) if x is not None else '' for x in move]
                writer.writerow(move_flat + [reward])
        print("Game data appended to CSV")

    def load_csv_data(self):
        if os.path.exists(CSV_FILE):
            X = []
            y = []
            with open(CSV_FILE, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    move = [x if x != '' else None for x in row[:-1]]

                    if len(move) != 9:
                        print(f"Invalid move data: {move}. Skipping...")
                        continue
                    reward = float(row[-1])

                    try:
                        move_encoded = self.preprocess_board(move)
                    except ValueError as e:
                        print(f"Error encoding move: {move}. Error: {e}")
                        continue
                    X.append(move_encoded)

                    target = np.zeros(9)
                    target[np.argmax(move_encoded.reshape(-1))] = reward
                    y.append(target)

            if X:
                self.fit_kmeans(X, n_clusters=5)
                X = np.array(X)
                y = np.array(y)
                X = np.expand_dims(X, axis=0)
                y = np.expand_dims(y, axis=0)

                print("Training model with CSV data...")
                self.model.fit(X, y, epochs=300, verbose=1)  # Increased epochs to 300


# Tic-Tac-Toe board with GUI
class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic Tac Toe")

        self.master.geometry('600x750')  # Wider window size
        self.master.configure(bg='#1e1e1e')  # Dark background

        self.game = TicTacToe()
        self.agent = CNNLSTMAgent()
        self.agent.load_model()

        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        button_style = {'font': 'Arial 40 bold', 'bg': '#333333', 'fg': '#87CEEB', 'relief': 'raised'}
        self.buttons = [tk.Button(self.master, text='', **button_style, command=lambda i=i: self.on_click(i)) for i in range(9)]

        for i, button in enumerate(self.buttons):
            button.grid(row=i // 3, column=i % 3, padx=5, pady=5, sticky="nsew")

        for i in range(3):
            self.master.grid_rowconfigure(i, weight=1, uniform="row")
            self.master.grid_columnconfigure(i, weight=1, uniform="col")

        self.reset_button = tk.Button(self.master, text="Reset", font='Arial 20 bold', command=self.reset_game, bg='#444444', fg='#87CEEB', relief='raised')
        self.reset_button.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.status_label = tk.Label(self.master, text="AI's turn!", font='Arial 30 bold', bg='#1e1e1e', fg='#87CEEB')
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.train_button = tk.Button(self.master, text="Train Model", font='Arial 20 bold', command=self.train_model, bg='#444444', fg='#87CEEB', relief='raised')
        self.train_button.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.win_loss_label = tk.Label(self.master, text="Wins: 0 | Losses: 0 | Draws: 0", font='Arial 20 bold', bg='#1e1e1e', fg='#87CEEB')
        self.win_loss_label.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.human_vs_ai_button = tk.Button(self.master, text="Human vs AI", font='Arial 20 bold', command=self.set_human_vs_ai_mode, bg='#444444', fg='#87CEEB', relief='raised')
        self.human_vs_ai_button.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.ai_vs_ai_button = tk.Button(self.master, text="AI vs AI", font='Arial 20 bold', command=self.set_ai_vs_ai_mode, bg='#444444', fg='#87CEEB', relief='raised')
        self.ai_vs_ai_button.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=5, pady=10)

        self.ai_vs_ai_mode = False

        self.master.grid_rowconfigure(3, weight=1, uniform="row")
        self.master.grid_rowconfigure(4, weight=1, uniform="row")
        self.master.grid_rowconfigure(5, weight=1, uniform="row")
        self.master.grid_rowconfigure(6, weight=1, uniform="row")
        self.master.grid_rowconfigure(7, weight=1, uniform="row")
        self.master.grid_rowconfigure(8, weight=1, uniform="row")

        self.reset_game()

    def set_human_vs_ai_mode(self):
        self.ai_vs_ai_mode = False
        self.reset_game()

    def set_ai_vs_ai_mode(self):
        self.ai_vs_ai_mode = True
        self.reset_game()

    def on_click(self, index):
        if not self.ai_vs_ai_mode and self.game.board[index] is None:
            print(f"Player {self.game.current_player} clicked on index {index}")
            self.game.make_move(index)
            self.update_board()
            winner = self.game.check_winner()

            if winner:
                self.end_game(winner)
            else:
                self.game.switch_player()
                self.status_label.config(text="AI's turn..." if self.game.current_player == 'O' else "Your turn!")
                if self.game.current_player == 'O':
                    self.master.after(500, self.ai_move)

    def ai_move(self):
        available_moves = [i for i, cell in enumerate(self.game.board) if cell is None]
        print(f"AI is thinking... Available moves: {available_moves}")
        move = self.agent.choose_action(self.game.board, available_moves)
        print(f"AI chose move: {move}")
        self.game.make_move(move)
        self.update_board()
        winner = self.game.check_winner()

        if winner:
            self.end_game(winner)
        else:
            self.game.switch_player()
            if self.ai_vs_ai_mode:
                self.master.after(500, self.ai_move)
            else:
                self.status_label.config(text="Your turn!")

    def update_board(self):
        for i, cell in enumerate(self.game.board):
            if cell == 'X' or cell == 'O':
                self.buttons[i].config(text=cell, fg='#87CEEB')
            else:
                self.buttons[i].config(text='')

    def end_game(self, winner):
        print(f"Game ended with winner: {winner}")
        if winner == 'Draw':
            self.draw_count += 1
            reward = 0.5  # Smaller reward for a draw
            self.status_label.config(text="It's a draw!")
        elif winner == 'X':
            self.win_count += 1
            reward = -1  # Smaller penalty for AI losing
            self.status_label.config(text="You win!" if not self.ai_vs_ai_mode else "AI 1 wins!")
        else:
            self.loss_count += 1
            reward = 1  # Smaller reward for AI winning
            self.status_label.config(text="AI wins!" if not self.ai_vs_ai_mode else "AI 2 wins!")

        print(f"Updating model with reward: {reward}")
        self.agent.update_model(self.game.move_sequence, reward)
        self.agent.update_epsilon()  # Decay epsilon after each game
        self.agent.save_model()
        self.agent.append_to_csv(self.game.move_sequence, reward)

        self.win_loss_label.config(text=f"Wins: {self.win_count} | Losses: {self.loss_count} | Draws: {self.draw_count}")

        if self.ai_vs_ai_mode:
            self.master.after(2000, self.reset_game)
        else:
            self.master.after(2000, self.reset_game)

    def reset_game(self):
        print("Resetting the game...")
        self.game.reset()
        self.update_board()

        if self.ai_vs_ai_mode:
            self.status_label.config(text="AI 1's turn!")
            self.master.after(500, self.ai_move)
        else:
            if self.game.current_player == 'O':
                self.status_label.config(text="AI's turn!")
                self.master.after(500, self.ai_move)
            else:
                self.status_label.config(text="Your turn!")

    def train_model(self):
        print("Retraining the model with CSV data...")
        self.agent.load_csv_data()
        self.status_label.config(text="Model retrained!")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
