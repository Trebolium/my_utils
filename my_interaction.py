def binary_answer():
    answered = False
    while not answered:
        print('Please type your answer as \'y\' or \'n\': ')
        answer = input()
        if answer == 'y':
            answered = True
            state = True
        elif answer == 'n':
            answered = True
            state = False
        else:
            print('Invalid answer. Please type a lowercase \'y\' or \'n\'')
            pass
    return state