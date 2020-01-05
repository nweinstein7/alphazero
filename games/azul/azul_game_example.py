class AzulGame(object):
    def __init__(self, number_of_players, number_of_factories):
        self.number_of_players = number_of_players
        self.number_of_factories = number_of_factories

    def __repr__(self):
        return "{} player game".format(self.number_of_players)


two_player_game = AzulGame(number_of_players=2, number_of_factories=5)
print("Azul game has {} players".format(two_player_game.number_of_players))

three_player_game = AzulGame(number_of_players=3, number_of_factories=7)
four_player_game = AzulGame(number_of_players=4, number_of_factories=9)

all_games = [three_player_game, four_player_game, two_player_game]

sorted_by_players = sorted(all_games, key=lambda game: game.number_of_players)
print(sorted_by_players)