"""Hand-curated seed catalog: well-known tracks across genres.

Used to build a real-track demo dataset by querying the iTunes Search API for
each (artist, title). Picked for recognizability — when the model recommends
similar tracks, you should immediately see whether the result makes sense.

Format: list of (artist, title, genre) tuples.
"""

SEED_TRACKS: list[tuple[str, str, str]] = [
    # --- Pop ---
    ("Taylor Swift", "Anti-Hero", "Pop"),
    ("Taylor Swift", "Blank Space", "Pop"),
    ("Olivia Rodrigo", "drivers license", "Pop"),
    ("Olivia Rodrigo", "good 4 u", "Pop"),
    ("Dua Lipa", "Levitating", "Pop"),
    ("Dua Lipa", "Don't Start Now", "Pop"),
    ("Harry Styles", "As It Was", "Pop"),
    ("Harry Styles", "Watermelon Sugar", "Pop"),
    ("Ed Sheeran", "Shape of You", "Pop"),
    ("Ed Sheeran", "Perfect", "Pop"),
    ("Billie Eilish", "bad guy", "Pop"),
    ("Billie Eilish", "Happier Than Ever", "Pop"),
    ("Ariana Grande", "thank u, next", "Pop"),
    ("Ariana Grande", "7 rings", "Pop"),
    ("The Weeknd", "Blinding Lights", "Pop"),

    # --- Hip-Hop ---
    ("Kendrick Lamar", "HUMBLE.", "Hip-Hop"),
    ("Kendrick Lamar", "Money Trees", "Hip-Hop"),
    ("Drake", "God's Plan", "Hip-Hop"),
    ("Drake", "Hotline Bling", "Hip-Hop"),
    ("Travis Scott", "SICKO MODE", "Hip-Hop"),
    ("Travis Scott", "goosebumps", "Hip-Hop"),
    ("J. Cole", "No Role Modelz", "Hip-Hop"),
    ("J. Cole", "MIDDLE CHILD", "Hip-Hop"),
    ("Jay-Z", "99 Problems", "Hip-Hop"),
    ("Eminem", "Lose Yourself", "Hip-Hop"),
    ("Eminem", "Without Me", "Hip-Hop"),
    ("Tyler, The Creator", "EARFQUAKE", "Hip-Hop"),
    ("Tyler, The Creator", "See You Again", "Hip-Hop"),
    ("Future", "Mask Off", "Hip-Hop"),
    ("Lil Wayne", "A Milli", "Hip-Hop"),

    # --- R&B ---
    ("Frank Ocean", "Thinkin Bout You", "R&B"),
    ("Frank Ocean", "Pyramids", "R&B"),
    ("SZA", "Good Days", "R&B"),
    ("SZA", "Kill Bill", "R&B"),
    ("The Weeknd", "Starboy", "R&B"),
    ("The Weeknd", "The Hills", "R&B"),
    ("Beyoncé", "Crazy in Love", "R&B"),
    ("Beyoncé", "Halo", "R&B"),
    ("H.E.R.", "Best Part", "R&B"),
    ("Daniel Caesar", "Best Part", "R&B"),
    ("Bruno Mars", "Locked Out of Heaven", "R&B"),
    ("Alicia Keys", "Fallin'", "R&B"),
    ("John Legend", "All of Me", "R&B"),

    # --- Rock ---
    ("Led Zeppelin", "Stairway to Heaven", "Rock"),
    ("Led Zeppelin", "Whole Lotta Love", "Rock"),
    ("Queen", "Bohemian Rhapsody", "Rock"),
    ("Queen", "Don't Stop Me Now", "Rock"),
    ("The Rolling Stones", "Paint It, Black", "Rock"),
    ("The Beatles", "Hey Jude", "Rock"),
    ("The Beatles", "Come Together", "Rock"),
    ("Pink Floyd", "Money", "Rock"),
    ("Pink Floyd", "Time", "Rock"),
    ("Nirvana", "Smells Like Teen Spirit", "Rock"),
    ("Foo Fighters", "Everlong", "Rock"),
    ("AC/DC", "Back in Black", "Rock"),
    ("Guns N' Roses", "Sweet Child O' Mine", "Rock"),

    # --- Indie / Alternative ---
    ("Arctic Monkeys", "Do I Wanna Know?", "Indie"),
    ("Arctic Monkeys", "505", "Indie"),
    ("The Strokes", "Last Nite", "Indie"),
    ("Tame Impala", "The Less I Know The Better", "Indie"),
    ("Tame Impala", "Borderline", "Indie"),
    ("Vampire Weekend", "A-Punk", "Indie"),
    ("Mac DeMarco", "Chamber of Reflection", "Indie"),
    ("Beach House", "Space Song", "Indie"),
    ("Phoebe Bridgers", "Motion Sickness", "Indie"),
    ("The 1975", "Somebody Else", "Indie"),
    ("Mac Miller", "Self Care", "Indie"),
    ("Cigarettes After Sex", "Apocalypse", "Indie"),

    # --- Country ---
    ("Johnny Cash", "Ring of Fire", "Country"),
    ("Johnny Cash", "Hurt", "Country"),
    ("Dolly Parton", "Jolene", "Country"),
    ("Dolly Parton", "9 to 5", "Country"),
    ("Garth Brooks", "Friends in Low Places", "Country"),
    ("Luke Combs", "Beautiful Crazy", "Country"),
    ("Morgan Wallen", "Whiskey Glasses", "Country"),
    ("Chris Stapleton", "Tennessee Whiskey", "Country"),
    ("Kacey Musgraves", "Rainbow", "Country"),
    ("Zach Bryan", "Something in the Orange", "Country"),

    # --- Electronic / EDM ---
    ("Daft Punk", "Get Lucky", "Electronic"),
    ("Daft Punk", "One More Time", "Electronic"),
    ("Calvin Harris", "Summer", "Electronic"),
    ("Calvin Harris", "Feel So Close", "Electronic"),
    ("Avicii", "Wake Me Up", "Electronic"),
    ("Avicii", "Levels", "Electronic"),
    ("The Chainsmokers", "Closer", "Electronic"),
    ("Skrillex", "Bangarang", "Electronic"),
    ("Deadmau5", "Strobe", "Electronic"),
    ("Disclosure", "Latch", "Electronic"),
    ("Flume", "Never Be Like You", "Electronic"),

    # --- Jazz ---
    ("Miles Davis", "So What", "Jazz"),
    ("John Coltrane", "Giant Steps", "Jazz"),
    ("Dave Brubeck", "Take Five", "Jazz"),
    ("Louis Armstrong", "What a Wonderful World", "Jazz"),
    ("Nina Simone", "Feeling Good", "Jazz"),
    ("Ella Fitzgerald", "Summertime", "Jazz"),
    ("Chet Baker", "My Funny Valentine", "Jazz"),
    ("Herbie Hancock", "Cantaloupe Island", "Jazz"),
    ("Thelonious Monk", "Round Midnight", "Jazz"),
    ("Duke Ellington", "Take the A Train", "Jazz"),

    # --- Classical ---
    ("Ludwig van Beethoven", "Symphony No. 5", "Classical"),
    ("Wolfgang Amadeus Mozart", "Eine kleine Nachtmusik", "Classical"),
    ("Johann Sebastian Bach", "Air on the G String", "Classical"),
    ("Frédéric Chopin", "Nocturne Op. 9 No. 2", "Classical"),
    ("Claude Debussy", "Clair de Lune", "Classical"),
    ("Pyotr Ilyich Tchaikovsky", "Swan Lake", "Classical"),
    ("Antonio Vivaldi", "Four Seasons - Spring", "Classical"),
    ("Erik Satie", "Gymnopédie No. 1", "Classical"),
    ("Ludovico Einaudi", "Nuvole Bianche", "Classical"),
    ("Max Richter", "On the Nature of Daylight", "Classical"),

    # --- Latin ---
    ("Bad Bunny", "Tití Me Preguntó", "Latin"),
    ("Bad Bunny", "Me Porto Bonito", "Latin"),
    ("J Balvin", "Mi Gente", "Latin"),
    ("Luis Fonsi", "Despacito", "Latin"),
    ("Shakira", "Hips Don't Lie", "Latin"),
    ("Shakira", "Waka Waka", "Latin"),
    ("Rosalía", "MALAMENTE", "Latin"),
    ("Karol G", "TUSA", "Latin"),
    ("Maluma", "Felices los 4", "Latin"),
    ("Ozuna", "Taki Taki", "Latin"),

    # --- Metal ---
    ("Metallica", "Enter Sandman", "Metal"),
    ("Metallica", "Master of Puppets", "Metal"),
    ("Black Sabbath", "Iron Man", "Metal"),
    ("Iron Maiden", "The Trooper", "Metal"),
    ("Slipknot", "Duality", "Metal"),
    ("System of a Down", "Chop Suey!", "Metal"),
    ("Tool", "Schism", "Metal"),
    ("Pantera", "Walk", "Metal"),
    ("Megadeth", "Symphony of Destruction", "Metal"),

    # --- Folk ---
    ("Bob Dylan", "Like a Rolling Stone", "Folk"),
    ("Bob Dylan", "Blowin' in the Wind", "Folk"),
    ("Simon & Garfunkel", "The Sound of Silence", "Folk"),
    ("Joni Mitchell", "Big Yellow Taxi", "Folk"),
    ("Mumford & Sons", "Little Lion Man", "Folk"),
    ("The Lumineers", "Ho Hey", "Folk"),
    ("Fleet Foxes", "White Winter Hymnal", "Folk"),
    ("Iron & Wine", "Such Great Heights", "Folk"),
    ("Bon Iver", "Skinny Love", "Folk"),
    ("Sufjan Stevens", "Mystery of Love", "Folk"),
]
