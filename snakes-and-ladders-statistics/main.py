import sqlite3
from random import randint

# import time

winner_position = 36

# constrains
ladders = {"3": 16, "5": 7, "15": 25, "18": 20, "21": 32}
snakes = {"12": 2, "14": 11, "17": 4, "31": 19, "35": 22}

# specify case
ladder_with_50_percent_to_take_it = False


class Player:
    def __init__(self):
        self.position = 1
        self.is_winner = False
        self.first_snake_immunity = False
        self.active_ladders = 0
        self.active_snakes = 0

    # function to move the player
    def play(self):
        face = randint(1, 6)  # roll the dice

        # time.sleep(0.5)
        # print(f'Begin pos {self.position} rolled {face}', end=' -> ')

        next_move = self.position + face
        self.is_winner = next_move == winner_position
        if next_move <= winner_position:
            self.position = next_move
            self.check_constrains()

        # print(f'Final pos {self.position}')
        # print()
        return self.is_winner

    # function to check if the player is in a ladder or a snake
    def check_constrains(self):
        key = str(self.position)

        if key in ladders:
            self.active_ladders += 1
            if not ladder_with_50_percent_to_take_it or (
                ladder_with_50_percent_to_take_it and randint(0, 1)
            ):
                self.position = ladders[key]
        elif key in snakes:
            self.active_snakes += 1
            if not self.first_snake_immunity:
                self.position = snakes[key]
            else:
                self.first_snake_immunity = False


# connecting to db
conn = sqlite3.connect("snakes-and-ladders/database.db")
cur = conn.cursor()

table = "Matches_with_player_2_start_on_square_3"
# Matches_with_player_2_start_on_square_2
# Matches_with_player_2_start_on_square_3
# Matches_with_player_2_start_on_square_4
# Matches_with_player_2_start_on_square_5
# Matches_with_player_2_first_snake_immunity
# Matches_with_50_percent_to_take_ladder

# creating table
sql_create = f"""CREATE TABLE IF NOT EXISTS {table}
                    (id              INTEGER NOT NULL PRIMARY KEY,
                    winner           INTEGER NOT NULL,
                    active_ladders   INTEGER NOT NULL,
                    active_snakes    INTEGER NOT NULL,
                    rolls            INTEGER NOT NULL)"""

cur.execute(sql_create)
conn.commit()

# query to insert data
sql_insert = f"""INSERT INTO {table}
                    (winner, active_ladders, active_snakes, rolls)
                VALUES
                    (?, ?, ?, ?)"""

# simulation
simulations = 10000
for i in range(simulations):
    # players
    player_1 = Player()
    player_2 = Player()
    player_2.position = 3
    # player_2.first_snake_immunity = True

    # states
    player_1_turn = True
    winner = False
    rolls = 0

    # running the game
    while not winner:
        # print(f'Player {1 if player_1_turn else 2}:')
        rolls += 1
        winner = player_1.play() if player_1_turn else player_2.play()
        player_1_turn = not player_1_turn

    winner = 1 if player_1.is_winner else 2
    active_ladders = player_1.active_ladders + player_2.active_ladders
    active_snakes = player_1.active_snakes + player_2.active_snakes

    # inserting data
    cur.execute(sql_insert, (winner, active_ladders, active_snakes, rolls))
    conn.commit()

# closing db
conn.close()
