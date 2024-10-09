import tkinter as tk
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
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
            self.move_sequence.append(self.board[:])  # Log board state for LSTM learning
            return True
        return False

    def check_winner(self):
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

    def reset(self):
        self.board = [None] * 9
        self.move_sequence = []  # Reset move sequence
        self.current_player = random.choice(['X', 'O'])  # Randomly select who starts


# AI agent for the game using LSTM only
class LSTMAgent:
    def __init__(self):
        self.model = self.build_model()
        self.encoder = OneHotEncoder(sparse_output=False)  # Fit encoder to the possible board states
        self.encoder.fit(np.array([['X'], ['O'], [None]]))

        # Epsilon-greedy strategy parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.99

        self.load_csv_data()

    def build_model(self):
        model = Sequential()

        # LSTM for processing the sequence of board states
        model.add(LSTM(128, input_shape=(None, 9), return_sequences=False))

        # Fully connected layer for choosing actions
        model.add(Dense(9, activation='softmax'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def preprocess_board(self, board):
        # One-hot encode the board and flatten it into a 1D array
        if len(board) != 9:
            raise ValueError(f"Invalid board length: {len(board)}. Expected 9 elements.")

        board_reshaped = np.array(board).reshape(-1, 1)
        board_encoded = self.encoder.transform(board_reshaped).reshape(9, 3)
        return board_encoded[:, 0]  # Use the one-hot encoded first dimension

    def choose_action(self, board, available_actions):
        # Epsilon-greedy strategy for exploration and exploitation
        if np.random.rand() <= self.epsilon:
            return random.choice(available_actions)
        else:
            # Preprocess the current board
            board_encoded = self.preprocess_board(board)
            board_encoded = np.expand_dims(board_encoded, axis=0)  # Add batch dimension
            board_encoded = np.expand_dims(board_encoded, axis=0)  # Add time step dimension

            # Predict the best move using LSTM
            predictions = self.model.predict(board_encoded)[0]

            # Choose the best available move
            available_predictions = [(i, predictions[i]) for i in available_actions]
            move = max(available_predictions, key=lambda x: x[1])[0]
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

            move_targets = np.zeros(9)
            move_targets[np.argmax(board_encoded)] = reward

            y.append(move_targets)

        X = np.array(X)
        y = np.array(y)

        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)

        self.model.fit(X, y, epochs=300, verbose=1)

    def save_model(self):
        self.model.save("tic_tac_toe_lstm.h5")

    def load_model(self):
        if os.path.exists("tic_tac_toe_lstm.h5"):
            self.model = tf.keras.models.load_model("tic_tac_toe_lstm.h5", compile=False)
            self.model.compile(optimizer=Adam(), loss='mse')

    def append_to_csv(self, move_sequence, reward):
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            for move in move_sequence:
                move_flat = [str(x) if x is not None else '' for x in move]
                writer.writerow(move_flat + [reward])

    def load_csv_data(self):
        if os.path.exists(CSV_FILE):
            X = []
            y = []
            with open(CSV_FILE, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    move = [x if x != '' else None for x in row[:-1]]

                    if len(move) != 9:
                        continue
                    reward = float(row[-1])

                    try:
                        move_encoded = self.preprocess_board(move)
                    except ValueError:
                        continue

                    X.append(move_encoded)

                    target = np.zeros(9)
                    target[np.argmax(move_encoded)] = reward
                    y.append(target)

            if X:
                X = np.array(X)
                y = np.array(y)
                X = np.expand_dims(X, axis=0)
                y = np.expand_dims(y, axis=0)

                self.model.fit(X, y, epochs=300, verbose=1)


# Tic-Tac-Toe board with GUI
class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic Tac Toe")

        # Modern color palette
        self.bg_color = "#2E2E2E"
        self.button_color = "#505050"
        self.text_color = "#F0F0F0"
        self.accent_color = "#42C6A3"
        self.highlight_color = "#FF6B6B"
        self.button_text_color = "#000000"  # Darker color for button text

        self.master.geometry('600x750')
        self.master.configure(bg=self.bg_color)

        self.game = TicTacToe()
        self.agent = LSTMAgent()
        self.agent.load_model()

        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        button_style = {
            'font': 'Helvetica 40 bold',
            'bg': self.button_color,
            'fg': self.button_text_color,  # Changed to darker color
            'activebackground': self.accent_color,
            'relief': 'flat',
            'bd': 0
        }
        self.buttons = [tk.Button(self.master, text='', **button_style, command=lambda i=i: self.on_click(i)) for i in range(9)]

        for i, button in enumerate(self.buttons):
            button.grid(row=i // 3, column=i % 3, padx=10, pady=10, sticky="nsew")

        for i in range(3):
            self.master.grid_rowconfigure(i, weight=1, uniform="row")
            self.master.grid_columnconfigure(i, weight=1, uniform="col")

        self.reset_button = tk.Button(self.master, text="Reset", font='Helvetica 20 bold', command=self.reset_game, bg=self.accent_color, fg=self.bg_color, relief='flat')
        self.reset_button.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.status_label = tk.Label(self.master, text="AI's turn!", font='Helvetica 30 bold', bg=self.bg_color, fg=self.accent_color)
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.train_button = tk.Button(self.master, text="Train Model", font='Helvetica 20 bold', command=self.train_model, bg=self.highlight_color, fg=self.bg_color, relief='flat')
        self.train_button.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.win_loss_label = tk.Label(self.master, text="Wins: 0 | Losses: 0 | Draws: 0", font='Helvetica 20 bold', bg=self.bg_color, fg=self.text_color)
        self.win_loss_label.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.human_vs_ai_button = tk.Button(self.master, text="Human vs AI", font='Helvetica 20 bold', command=self.set_human_vs_ai_mode, bg=self.button_color, fg=self.button_text_color, relief='flat')
        self.human_vs_ai_button.grid(row=7, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        self.ai_vs_ai_button = tk.Button(self.master, text="AI vs AI", font='Helvetica 20 bold', command=self.set_ai_vs_ai_mode, bg=self.button_color, fg=self.button_text_color, relief='flat')
        self.ai_vs_ai_button.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

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
        move = self.agent.choose_action(self.game.board, available_moves)
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
                self.buttons[i].config(text=cell, fg=self.accent_color)
            else:
                self.buttons[i].config(text='')

    def end_game(self, winner):
        if winner == 'Draw':
            self.draw_count += 1
            reward = 0.5
            self.status_label.config(text="It's a draw!")
        elif winner == 'X':
            self.win_count += 1
            reward = -1
            self.status_label.config(text="You win!" if not self.ai_vs_ai_mode else "AI 1 wins!")
        else:
            self.loss_count += 1
            reward = 1
            self.status_label.config(text="AI wins!" if not self.ai_vs_ai_mode else "AI 2 wins!")

        self.agent.update_model(self.game.move_sequence, reward)
        self.agent.update_epsilon()
        self.agent.save_model()
        self.agent.append_to_csv(self.game.move_sequence, reward)

        self.win_loss_label.config(text=f"Wins: {self.win_count} | Losses: {self.loss_count} | Draws: {self.draw_count}")

        if self.ai_vs_ai_mode:
            self.master.after(2000, self.reset_game)
        else:
            self.master.after(2000, self.reset_game)

    def reset_game(self):
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
        self.agent.load_csv_data()
        self.status_label.config(text="Model retrained!")


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()
