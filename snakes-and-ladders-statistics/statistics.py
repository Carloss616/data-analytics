import sqlite3


class Database:
    def __init__(self, database: str):
        self.conn = sqlite3.connect(f"snakes-and-ladders/{database}.db")
        self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def total(self, table: str):
        # getting the total matches
        self.cur.execute(f"SELECT COUNT(*) FROM {table}")
        return self.cur.fetchone()[0]

    def winner_1_count(self, table: str):
        # getting the count of winner 1
        self.cur.execute(f"SELECT COUNT(*) FROM {table} WHERE winner = 1")
        return self.cur.fetchone()[0]

    def active_snakes(self, table: str):
        # getting list of active snakes
        self.cur.execute(f"SELECT active_snakes FROM {table}")
        return [value[0] for value in self.cur.fetchall()]

    def rolls(self, table: str):
        # getting list of rolls
        self.cur.execute(f"SELECT rolls FROM {table}")
        return [value[0] for value in self.cur.fetchall()]


def get_average_by_square(db: Database, square):
    table = f"Matches_with_player_2_start_on_square_{square}"
    average = db.winner_1_count(table) * 100 / db.total(table)
    print(f"    |- Square {square}: {average:.2f}")


# connecting to db
db = Database("database")

table = "Matches"
total = db.total(table)

# 1. In a two-person game, what is the probability that the player who starts the game wins?
winner_1_count = db.winner_1_count(table)
average = winner_1_count * 100 / total
print(f"1. The probability that the player who starts the game wins is {average:.2f}%.")

# 2. On average, how many snakes are landed on in each game?
active_snakes = db.active_snakes(table)
average = sum(active_snakes) / len(active_snakes)
print(f"2. The average number of the snakes are landed on in each game is {average:.2f}.")

# 3. If each time a player landed on a ladder and there was only a 50 % chance they could take it,
# what is the average number of rolls needed to complete a game?
table = "Matches_with_50_percent_to_take_ladder"
rolls = db.rolls(table)
average = sum(rolls) / len(rolls)
print(f"3. The average number of rolls needed to complete a game is {average:.2f}.")

# 4. Starting with the base game, you decide you want the game to have approximately fair odds.
# You do this by changing the square that Player 2 starts on. Which square for Player 2’s start
# position gives the closest to equal odds for both players?
print("4. The probability that the player 1 wins by player 2 start on")
for square in range(2, 11):
    get_average_by_square(db, square)
print("   The square with the closest to equal odds for both players is 8.")

# 5. In a different attempt to change the odds of the game, instead of starting Player 2 on a
# different square, you decide to give Player 2 immunity to the first snake that they land on.
# What is the approximate probability that Player 1 wins now?
table = "Matches_with_player_2_first_snake_immunity"
average = db.winner_1_count(table) * 100 / db.total(table)
print(f"5. The probability that the player 1 wins is {average:.2f}%.")

# closing db
del db
