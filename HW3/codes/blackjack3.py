import numpy as np
import matplotlib as mpl
import pandas as pd
import random 
import math


HIT = 1
STICK = 0 



class BlackJack:
    def __init__(self):
        
        self.actions = [STICK,HIT]
        self.num_states = 22
        self.player_policy = np.ones(self.num_states, dtype = np.int)
        #self.init_player_policy()
        self.dealer_policy = np.ones(self.num_states, dtype = np.int)
        self.initPDPolicies()
        self.num_cards = 13
        self.face_card_val = 10
        
        
    def initPDPolicies(self):
        
        #initializing player policy
        for i in range(12, 20):
            self.player_policy[i] = self.actions[1]
        self.player_policy[20],self.player_policy[21] = self.actions[0],self.actions[0]
        
        #initializing dealer policy
        for i in range(12, 17):
            self.dealer_policy[i] = self.actions[1]
        for i in range(17, 22):
            self.dealer_policy[i] = self.actions[0]
    
        
    #get action corrsponding to state    
    def getPlayerPolicySA(self, player_state):
        return self.player_policy[player_state]
    

    # function form of behavior policy of player
    def getPlayerBehavPolicySA(self):
        if np.random.binomial(1, 0.5) == 1:
            return self.actions[1]
        return self.actions[0]

    def drawCardAndValue(self):
        
        random_card = min(np.random.randint(1, self.num_cards+1),self.face_card_val)
        card_val = 11 if random_card == 1 else random_card

        return random_card, card_val
    
    
    def getCardValue(self,card):
        card_val = 11 if card == 1 else card
        return card_val
    
    
    def setGameInitStatus(self,init_state, player_state,pusable_ace_flag,dealer_card1,dealer_card2,dusable_ace_flag):

        if init_state is None:
            #generate random initial state
            while player_state < 12:
                #if player' sum is less than 12, always hit
                card, card_value = self.drawCardAndValue()
                player_state += card_value

                # if the player sum > 21, he may hold one or two aces.
                if player_state > 21:
                    # last card must be ace, so use it as 1 than 11(usable)
                    player_state -= 10
                else:
                    pusable_ace_flag |= (1 == card)

            #initialize cards of dealer, suppose dealer will show the first card he gets
            dealer_card1, dealer_card1_val = self.drawCardAndValue()
            dealer_card2, dealer_card2_val = self.drawCardAndValue()

        else:
            pusable_ace_flag, player_state, dealer_card1 = init_state
            dealer_card1_val = self.getCardValue(dealer_card1)
            dealer_card2, dealer_card2_val  = self.drawCardAndValue()

        #initial state of the game
        state = [pusable_ace_flag, player_state, dealer_card1]

        #initialize dealer state
        dealer_state = dealer_card1_val + dealer_card2_val
        dusable_ace_flag = 1 in (dealer_card1, dealer_card2)

        if dealer_state > 21:
            #since dealer_state>21, he holds two aces and 
            #hence use second ace as 1 than 11
            dealer_state -= 10

        return state, player_state, pusable_ace_flag, dealer_card1, dealer_card1_val, dealer_card2, dealer_card2_val, dealer_state, dusable_ace_flag



    def playerPlays(self,init_action,player_state,pusable_ace_flag,player_seq_plays,dealer_card1):

        while True:
            if init_action is not None:
                action = init_action
                init_action = None
            else:
                #get action based on current sum
                action = self.getPlayerBehavPolicySA()

            #track player's sequence of play
            player_seq_plays.append([(pusable_ace_flag, player_state, dealer_card1), action])

            #if player decides to stick then break
            if action == self.actions[0]:
                break

            #if player wants hit, then get the new card
            card, card_val = self.drawCardAndValue()

            #keeping track of the ace count whether 1 or 2 
            cnt_ace = int(pusable_ace_flag)
            if card == 1:
                cnt_ace += 1
            player_state += card_val

            #if the player has a usable ace, use it as 1 to avoid busting and continue.
            while player_state > 21 and cnt_ace:
                player_state -= 10
                cnt_ace -= 1

            #if player busts i.e. player_state > 21, then he loses and return immediately 
            if player_state > 21:
                return -1, player_seq_plays
            pusable_ace_flag = (cnt_ace == 1)

        return player_state, player_seq_plays


    def dealerPlays(self, dealer_state, dusable_ace_flag):
        while True:
            #get action based on dealer's state
            action = self.dealer_policy[dealer_state]

            if action == self.actions[0]:
                break

            # if hit, get a new card
            new_card, new_card_val = self.drawCardAndValue()
            dealer_state += new_card_val

            #keeping track of aces with the dealer
            cnt_ace = int(dusable_ace_flag)
            if new_card == 1:
                cnt_ace += 1

            #if the dealer has usable ace => val is 1 to avoid bust
            while dealer_state > 21 and cnt_ace:
                dealer_state -= 10
                cnt_ace -= 1

            #if dealer_state > 21, dealer busts
            if dealer_state > 21:
                return 1
            dusable_ace_flag = (cnt_ace == 1)

        return dealer_state


    
    def getActions(self):
        return self.actions
    

    
    def playGame(self, init_state = None, init_action = None):
        #initially player's status
        #player state, player sequence of plays, whether he has usable ace 
        player_state, player_seq_plays, pusable_ace_flag = 0 , [], False

        #dealer's initial cards, whether he has usable ace
        dealer_card1, dealer_card2, dusable_ace_flag = 0, 0, False

        state, player_state, pusable_ace_flag, dealer_card1, dealer_card1_val, dealer_card2, dealer_card2_val,\
        dealer_state, dusable_ace_flag = self.setGameInitStatus(init_state, player_state,pusable_ace_flag,\
                                                                      dealer_card1,dealer_card2,dusable_ace_flag)

        #we start the game where player plays first 
        player_state, player_seq_plays = self.playerPlays(init_action, player_state, pusable_ace_flag, player_seq_plays, dealer_card1)
        if player_state == -1:
            return state, player_state, player_seq_plays

        #then dealer plays
        dealer_state = self.dealerPlays(dealer_state, dusable_ace_flag)
        if player_state == 1:
            return state, dealer_state, player_seq_plays

        #compare the sum between player and dealer
        if player_state > dealer_state:
            return state, 1, player_seq_plays
        elif player_state == dealer_state:
            return state, 0, player_seq_plays
        else:
            return state, -1, player_seq_plays
        
        
    def playGame(self, policy, init_state = None, init_action = None):
        #initially player's status
        #player state, player sequence of plays, whether he has usable ace 
        player_state, player_seq_plays, pusable_ace_flag = 0 , [], False

        #dealer's initial cards, whether he has usable ace
        dealer_card1, dealer_card2, dusable_ace_flag = 0, 0, False

        state, player_state, pusable_ace_flag, dealer_card1, dealer_card1_val, dealer_card2, dealer_card2_val,\
        dealer_state, dusable_ace_flag = self.setGameInitStatus(init_state, player_state,pusable_ace_flag,\
                                                                      dealer_card1,dealer_card2,dusable_ace_flag)

        #we start the game where player plays first 
        player_state, player_seq_plays = self.playerPlays(init_action, player_state, pusable_ace_flag, player_seq_plays, dealer_card1)
        if player_state == -1:
            return state, player_state, player_seq_plays

        #then dealer plays
        dealer_state = self.dealerPlays(dealer_state, dusable_ace_flag)
        if player_state == 1:
            return state, dealer_state, player_seq_plays

        #compare the sum between player and dealer
        if player_state > dealer_state:
            return state, 1, player_seq_plays
        elif player_state == dealer_state:
            return state, 0, player_seq_plays
        else:
            return state, -1, player_seq_plays