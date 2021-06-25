SENTENCES = {
    'D1': 'I like big butts. I can not lie.',
    'D2': 'some other song.',

    '1': 'It was performed in 13 teams, and Al Arabi Kuwait won the championship.',
    '2': 'He made his debut for Rosenborg in 1999.',
    '3': 'She started singing as a toddler, considering Márta Sebestyén a role model.',
    '4': 'Central Radio launched on 25 September 2008.',
    '5': 'The 1877 City of Wellington by-election was a by-election held in the multi-member City of Wellington electorate during the 6th New Zealand Parliament, on 27 March 1877.',
}


def get_sentence():
    print('---')
    print('Dummy')
    for key, sent in SENTENCES.items():
        if not key.startswith('D'):
            continue
        print('(' + key + ') ' + sent)
    print('---')
    print('Wiki')
    for key, sent in SENTENCES.items():
        if key.startswith('D'):
            continue
        print('(' + key + ') ' + sent)
    print('---')

    sent = input('Pick a sentence or enter a custom one: ')
    if sent in ['exit', 'quit', 'q']:
        exit()

    if sent in SENTENCES:
        return SENTENCES[sent]

    return sent
