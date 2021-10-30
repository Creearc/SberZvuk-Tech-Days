
if True:
  arr = []
  arr1 = []
  f = open('vox1_meta.csv', 'r')
  for l in f:
    for word in l.split()[1].replace('_', ' ' ).split():
      arr.append(word.lower())
  f.close()

  arr = set(arr)
  for i in arr1:
      arr.add(i.lower())
  arr = list(arr)
  arr.sort()
  out = []
  k = 5
  print('arr = [')
  for i in range(len(arr) // k):
    print('''  "{}",'''.format('", "'.join([element for element in arr[i*k : (i+1)*k]])))
  print('''  "{}" '''.format('", "'.join([element for element in arr[(i+1)*k :]])))
  print('  ]')

arr2 = [
  "A.J.", "A.R.", "Aamir", "Aaron", "Abbie",
  "Abel", "Abigail", "Abraham", "Abrams", "Accola",
  "Ackles", "Adam", "Adler", "Adlon", "Adrianne",
  "Aduba", "Affleck", "Aghdashloo", "Agyeman", "Agyness",
  "Ahmed", "Aidan", "Ajay", "Akerman", "Akshay",
  "Alain", "Alan", "Alba", "Alda", "Aldis",
  "Alex", "Alexa", "Alexander", "Alexandra", "Alexz",
  "Alfonso", "Alfre", "Ali", "Alice", "Alicja",
  "Alison", "Allen", "Allison", "Almeida", "Alonso",
  "Amalric", "Amanda", "Amaury", "Amber", "Amell",
  "America", "Amitabh", "Amos", "Amy", "Ana",
  "Anderson", "Andre", "Andrea", "Andrew", "Andrews",
  "Andy", "Aneurin", "Ang", "Angela", "Ann",
  "Annable", "Anne", "Ansel", "Anthony", "Anton",
  "Antonio", "Appleby", "Ardant", "Arden", "Arianda",
  "Armand", "Armie", "Armstrong", "Arnaz", "Arnett",
  "Arngrim", "Arnold", "Arterton", "Asa", "Aselton",
  "Ashley", "Asner", "Assante", "Astin", "Atack",
  "Atias", "Atkinson", "Attenborough", "Auberjonois", "Audra",
  "Audrina", "Austin", "Auteuil", "Avan", "Avgeropoulos",
  "Aykroyd", "Ayoade", "B.J.", "Babett", "Baccarin",
  "Bachchan", "Bachleda", "Bacon", "Badge", "Badler",
  "Baker", "Bakula", "Balan", "Baldwin", "Balfe",
  "Bamford", "Barbara", "Barker", "Barks", "Barnard",
  "Barr", "Barrowman", "Barton", "Baruchel", "Bateman",
  "Bauer", "Bautista", "Baxter", "Bay", "Beach",
  "Beals", "Bean", "Bear", "Beaufort", "Becker",
  "Beckinsale", "Beharie", "Bell", "Bellamy", "Belmondo",
  "Ben", "Benanti", "Bennet", "Bentley", "Berg",
  "Berger", "Bergeron", "Berglund", "Bergman", "Berkoff",
  "Bernhard", "Besson", "Beth", "Bethany", "Betty",
  "Bibb", "Bichir", "Bier", "Bill", "Billie",
  "Bingbing", "Bjorlin", "Black", "Blackburn", "Blair",
  "Blake", "Bleibtreu", "Blessed", "Blethyn", "Bleu",
  "Bloodgood", "Blucas", "Bob", "Bobby", "Boldman",
  "Bolger", "Bomer", "Bonham", "Boniadi", "Bonneville",
  "Bonnie", "Booboo", "Booth", "Borgnine", "Borstein",
  "Boseman", "Bova", "Bowler", "Boxleitner", "Boyce",
  "Brad", "Bradley", "Braeden", "Braff", "Brar",
  "Braugher", "Breckin", "Brenda", "Brendan", "Breslin",
  "Brett", "Brewster", "Brian", "Bridget", "Brit",
  "Broadbent", "Brody", "Brolin", "Brook", "Brooke",
  "Brown", "Bruce", "Bruno", "Brydon", "Buckley",
  "Bure", "Buring", "Burke-Charvet", "Burstyn", "Burt",
  "Burton", "Busey", "Butterfield", "Buzolic", "C.",
  "C.K.", "CCH", "Cabrera", "Cain", "Caitriona",
  "Caity", "Callan", "Callie", "Callies", "Callow",
  "Cameron", "Campbell", "Canals-Barrera", "Candace", "Candice",
  "Canet", "Cannavale", "Cannon", "Capaldi", "Caplan",
  "Carano", "Cardellini", "Carell", "Carey", "Carlyle",
  "Carmichael", "Caroline", "Carolyn", "Carpenter", "Carrie",
  "Carroll", "Carson", "Carter", "Cartwright", "Casas",
  "Casey", "Cassandra", "Cassel", "Cassidy", "Caterina",
  "Catherine", "Caviezel", "Cavill", "Cedric", "Celia",
  "Cera", "Chabanol", "Chace", "Chadwick", "Chaplin",
  "Charice", "Charles", "Charlie", "Charlotte", "Chazz",
  "Chelsea", "Chen", "Cher", "Cheryl", "Chi",
  "Chiwetel", "Chloe", "Cho", "Chopra", "Chris",
  "Christian", "Christie", "Christina", "Christopher", "Chung",
  "Church", "Cierra", "Cilla", "Cillian", "Cindy",
  "Clarke", "Clarkson", "Clash", "Claudia", "Clay",
  "Cleary", "Cleese", "Clement", "Cliff", "Clive",
  "Close", "Clunes", "Cody", "Colbert", "Cole",
  "Coleman", "Colfer", "Colin", "Collins", "Colman",
  "Conlin", "Connery", "Connick", "Connor", "Considine",
  "Constance", "Cook", "Cooke", "Coolidge", "Cooper",
  "Copeland", "Copley", "Copon", "Coppola", "Corbett",
  "Corbin", "Corddry", "Corey", "Cornish", "Cory",
  "Cosgrove", "Cosmo", "Costas", "Coster-Waldau", "Cote",
  "Cotillard", "Courtney", "Cox", "Coyote", "Craig",
  "Crawford", "Crichton", "Criss", "Cristin", "Cromwell",
  "Crosby", "Cross", "Cruikshank", "Cruz", "Cucinotta",
  "Cudlitz", "Cumming", "Cummings", "Cupo", "Curran",
  "Curry", "Curtis", "Cusack", "Cyndi", "Cyrus",
  "Daddario", "Dae", "Dafoe", "Dakota", "Dale",
  "Daley", "Dallas", "Daly", "Damian", "Damon",
  "Dan", "Dana", "Dance", "Dancy", "Dane",
  "Danica", "Daniel", "Danielle", "Daniels", "Danny",
  "Dano", "Danson", "Darby", "Darren", "Davalos",
  "Dave", "Davenport", "Davern", "Davi", "David",
  "Davis", "Dawn", "Day", "Dayal", "De",
  "Dean", "Debbouze", "Debi", "Debra", "Dee",
  "Deidre", "Dekker", "Delany", "Della", "Delon",
  "Demian", "Dempsey", "Denis", "Denise", "Dennehy",
  "Dennings", "Derbez", "Derek", "Dermot", "Dern",
  "Devgn", "Devine", "Deyn", "DiMaggio", "Diablo",
  "Diamond", "Diane", "Dice", "Dick", "Dickinson",
  "Diego", "Diller", "Diogo", "Dockery", "Doherty",
  "Dohring", "Domhnall", "Dominic", "Don", "Donal",
  "Donald", "Donna", "Donnell", "Donnie", "Donovan",
  "Dormer", "Dot-Marie", "Douglas", "Dougray", "Doutzen",
  "Dove", "Downey", "Dr.", "Drake", "Dratch",
  "Dre", "Drea", "Dreama", "Drew", "Driver",
  "Duffy", "Duhamel", "Duncan", "Dunn", "Dustin",
  "Dutton", "Duty", "Dyan", "Dyer", "Dyke",
  "Dykstra", "Dylan", "Dyrdek", "E.", "Earl",
  "Earles", "Eartha", "Ebert", "Ed", "Eddie",
  "Edelstein", "Eden", "Edgar", "Edgerton", "Eduardo",
  "Edward", "Efren", "Efron", "Ehle", "Eisenberg",
  "Ejiofor", "Elaine", "Eleanor", "Elgort", "Eli",
  "Elisabeth", "Elise", "Elizabeth", "Elizondo", "Elle",
  "Ellen", "Elodie", "Emerson", "Emile", "Emily",
  "Emmanuel", "Emmerich", "Emraan", "Entertainer", "Eoin",
  "Epatha", "Eric", "Erik", "Erin", "Ernest",
  "Ernie", "Esai", "Esposito", "Estelle", "Estrada",
  "Eugene", "Eugenio", "Eva", "Evanna", "Evans",
  "Eve", "Ezarik", "Ezra", "Facinelli", "Faith",
  "Fallon", "Fanning", "Fanny", "Farahani", "Farrah",
  "Faustino", "Favino", "Favreau", "Fawad", "Fegan",
  "Feig", "Feldman", "Felicia", "Felicity", "Fergie",
  "Fernandez", "Ferrara", "Ferrera", "Ferres", "Ferrigno",
  "Feuerstein", "Fichtner", "Fielding", "Fillion", "Finley",
  "Finola", "Fiona", "Fischer", "Fisher", "Flanery",
  "Flanigan", "Flannery", "Florence", "Fogler", "Foley",
  "Ford", "Forest", "Forster", "Forte", "Fox",
  "Foxx", "Frain", "Francia", "Francis", "Frank",
  "Fred", "Freddie", "Freema", "Freida", "French",
  "Froggatt", "Fry", "Fugit", "Furlan", "Furtado",
  "Gad", "Gadon", "Gaiman", "Gainsbourg", "Galecki",
  "Galifianakis", "Gallagher", "Ganz", "Garcia", "Garfield",
  "Garlin", "Garofalo", "Garrett", "Garry", "Garson",
  "Gary", "Gaspard", "Gasteyer", "Gates", "Gatiss",
  "Gavankar", "Gavaris", "Gay", "Geena", "Gemma",
  "George", "Geri", "Gervais", "Gerwig", "Gevinson",
  "Giamatti", "Gibbs", "Gibson", "Giddish", "Gifford",
  "Gilbert", "Gilles", "Gillian", "Gilligan", "Gina",
  "Ginger", "Ginnifer", "Giovanni", "Giuliana", "Giuntoli",
  "Gleeson", "Glen", "Glenister", "Glenn", "Gless",
  "Gloria", "Glover", "Goggins", "Gold", "Goldblum",
  "Golshifteh", "Gonzalo", "Goode", "Goodman", "Goodwin",
  "Goran", "Gordon", "Gordon-Levitt", "Gouw", "Grabeel",
  "Graham", "Grainger", "Grant", "Gray", "Grazia",
  "Green", "Greene", "Greer", "Greg", "Gregg",
  "Greig", "Greta", "Griffin", "Griffo", "Groban",
  "Gross", "Grunberg", "Grylls", "Gugu", "Guillaume",
  "Guillory", "Gummer", "Gunn", "Guy", "Gwendoline",
  "Gyllenhaal", "Haddock", "Haden", "Hahn", "Hailee",
  "Hal", "Hale", "Hall", "Halliwell", "Hamill",
  "Hammer", "Hammond", "Handler", "Hannah", "Hanratty",
  "Hans", "Hansard", "Harden", "Hardin", "Hardwick",
  "Hardwicke", "Harewood", "Harmon", "Harper", "Harrelson",
  "Harris", "Harry", "Hart", "Harvey", "Hashmi",
  "Hassan", "Hatcher", "Hathaway", "Hauer", "Hawes",
  "Hawkes", "Hawkins", "Hayley", "Heather", "Hector",
  "Hedlund", "Heidi", "Heigl", "Heike", "Helena",
  "Hemingway", "Hemsworth", "Henao", "Henderson", "Hendricks",
  "Hendrix", "Henner", "Hennesy", "Henrie", "Henry",
  "Henson", "Herfurth", "Hershey", "Heughan", "Hiddleston",
  "Highmore", "Hilarie", "Hilary", "Hill", "Hilson",
  "Hirsch", "Hodge", "Hodgman", "Hoechlin", "Holliday",
  "Holt", "Hooda", "Hooper", "Hope", "Horan",
  "Horne", "Hoss", "Hough", "Howard", "Howie",
  "Hudgens", "Hudson", "Huertas", "Hugh", "Hughes",
  "Hunter", "Huntington-Whiteley", "Hurt", "Hye-kyo", "Hyland",
  "ID", "Iain", "Ian", "Ice", "Idina",
  "Ifans", "Imelda", "Imrie", "Inaba", "India",
  "Indira", "Ingrid", "Irina", "Irons", "Irrfan",
  "Irvine", "Isaac", "Isla", "Iwan", "Izabella",
  "Izzard", "Izzo", "J", "J.J.", "J.K.",
  "Jack", "Jackman", "Jackson", "Jacob", "Jacobi",
  "Jacobs", "Jacobson", "Jacqueline", "Jai", "Jaime",
  "Jake", "Jamel", "James", "Jamie", "Jamie-Lynn",
  "Jana", "Jane", "Janeane", "Janel", "Janet",
  "Janice", "Janina", "Janine", "January", "Jared",
  "Jasika", "Jasmine", "Jason", "Jay", "Jean",
  "Jean-Marc", "Jean-Paul", "Jeanine", "Jeannie", "Jeff",
  "Jefferies", "Jeffrey", "Jemaine", "Jen", "Jenifer",
  "Jenkins", "Jenna", "Jenner", "Jennette", "Jennifer",
  "Jenny", "Jensen", "Jeong", "Jeremy", "Jerry",
  "Jesse", "Jessica", "Jessie", "Jet", "Jill",
  "Jillette", "Jillian", "Jim", "Jimmy", "Jo",
  "Joan", "Joanna", "Joanne", "Joaquim", "Jobs",
  "Joe", "Joel", "Jogia", "John", "Johnathon",
  "Johnny", "Johnson", "Johnston", "Jon", "Jonathan",
  "Jones", "Joosten", "Jordan", "Jordana", "Jordin",
  "Jorge", "Jorja", "Joseph", "Josh", "Joss",
  "Jovovich", "Jr.", "Judith", "Judy", "Julia",
  "Julian", "Julianne", "Julie", "Justice", "Justine",
  "K.", "Kal", "Kaling", "Kanan", "Kane",
  "Kang", "Kangana", "Kapoor", "Karan", "Karl",
  "Karla", "Karoline", "Kassovitz", "Kat", "Kate",
  "Katee", "Katey", "Katharine", "Katherine", "Katheryn",
  "Kathie", "Kathleen", "Kathryn", "Kathy", "Katic",
  "Katie", "Katy", "Kay", "Kaya", "Kaye",
  "Kayvan", "Keach", "Keaton", "Keegan", "Keeley",
  "Keitel", "Kel", "Kellan", "Kelli", "Kelly",
  "Ken", "Kenan", "Kendra", "Kennedy", "Kenny",
  "Kenton", "Kenya", "Keri", "Kerr", "Kesha",
  "Kevin", "Khan", "Kid", "Kiefer", "Kiernan",
  "Kierston", "Kim", "Kimberly", "Kind", "Kingston",
  "Kinnaman", "Kinsey", "Kirsten", "Kitsch", "Kitt",
  "Klattenhoff", "Knight", "Knudsen", "Kodi", "Koechner",
  "Korey", "Kove", "Kramer", "Krasinski", "Kris",
  "Kristen", "Kristian", "Kriti", "Kroes", "Kroll",
  "Kross", "Kruger", "Krupa", "Krusiec", "Kudrow",
  "Kumar", "Kunis", "Kuno", "Kurylenko", "Kwanten",
  "Kyle", "Kylie", "Kym", "L.", "La",
  "Lacey", "Lachey", "Ladd", "Lafferty", "Lakshmi",
  "Lambert", "Lang", "Lanter", "Larroquette", "Lasance",
  "Lasseter", "Lathan", "Latimore", "Lauper", "Laura",
  "Laurie", "Lauryn", "Lautner", "Lavell", "Laverne",
  "Lavin", "Laz", "Lea", "Leakes", "Leary",
  "Leclerc", "Lee", "Lemmon", "Lena", "Lennon",
  "Leo", "Leone", "Leonor", "Lesley", "Leslie",
  "Leto", "Letterman", "Levi", "Levy", "Lewis",
  "Li", "Light", "Lillard", "Lilley", "Lily",
  "Lin", "Linda", "Lindelof", "Lindsay", "Lindsey",
  "Lipton", "Lisa", "Little", "Livingston", "Lizzy",
  "Lloyd", "Loaf", "Loan", "Logue", "Lolo",
  "Longoria", "Lopez", "Lorde", "Lorenza", "Loretta",
  "Lotte", "Lotz", "Lou", "Louis", "Louis-Dreyfus",
  "Lovitz", "Lowell", "Lowndes", "Luc", "Lucas",
  "Lucci", "Lucie", "Lucy", "Ludacris", "Luke",
  "Lumley", "Luna", "Lutz", "Lynch", "Lynn",
  "Lyons", "MacFarlane", "MacInnes", "MacLachlan", "Macchio",
  "Macdonald", "Macfadyen", "Macken", "Mackie", "Madden",
  "Mader", "Madsen", "Maggie", "Maher", "Mai",
  "Makatsch", "Malden", "Malek", "Malhotra", "Malin",
  "Mallika", "Mamet", "Mamie", "Mandel", "Mando",
  "Mandy", "Mandylor", "Mangan", "Manish", "Mann",
  "Manning", "Mansi", "Manson", "Mantooth", "Mantzoukas",
  "Manville", "Mara", "Marc", "Marcia", "Marcil",
  "Margo", "Margolyes", "Margot", "Maria", "Marie",
  "Mariel", "Marilu", "Marini", "Marino", "Mario",
  "Marion", "Marisa", "Mark", "Marla", "Marling",
  "Mars", "Marsden", "Marshall", "Martella", "Martin",
  "Martindale", "Martine", "Martinez", "Mary", "Mary-Louise",
  "Maslany", "Maslow", "Mason", "Mathieu", "Mathilda",
  "Matt", "Matteo", "Matthau", "Matthew", "Max",
  "May", "Maya", "Mayall", "Mazar", "Mazouz",
  "Mbatha-Raw", "McAdams", "McAuliffe", "McBride", "McCarthy",
  "McCartney", "McClanahan", "McClintock", "McClure", "McCormack",
  "McCormick", "McCurdy", "McCutcheon", "McDermott", "McDonald",
  "McDonnell", "McDonough", "McElhone", "McEntire", "McFadden",
  "McGann", "McHale", "McIver", "McKellar", "McKenzie",
  "McKinnon", "McLendon-Covey", "McNamara", "McPartlin", "McPhee",
  "McQueen", "McTeer", "Mears", "Meat", "Medeiros",
  "Megan", "Meghan", "Melissa", "Mellor", "Melody",
  "Melora", "Mena", "Menounos", "Menzel", "Merchant",
  "Meredith", "Merkerson", "Messina", "Metcalfe", "Meyer",
  "Meyers", "Mia", "Michael", "Michaela", "Michaels",
  "Michele", "Michelle", "Mila", "Miles", "Miley",
  "Milioti", "Milla", "Miller", "Milligan", "Mills",
  "Minaj", "Minchin", "Mindy", "Minogue", "Mintz-Plasse",
  "Mira", "Miranda", "Miriam", "Mischa", "Misha",
  "Mison", "Missi", "Mitchel", "Mitchell", "Modine",
  "Moffat", "Molly", "Momsen", "Monaco", "Monaghan",
  "Monahan", "Monica", "Montag", "Monteith", "Montgomery",
  "Moon", "Moore", "Morales", "Moran", "Morena",
  "Morgado", "Morgan", "Moritz", "Morrison", "Morrissey",
  "Morse", "Mortensen", "Moss", "Mota", "Mowry",
  "Mowry-Housley", "Moynahan", "Muhney", "Mulaney", "Mulgrew",
  "Mulroney", "Munn", "Murino", "Murphy", "Musso",
  "MyAnna", "Mya", "Nadia", "Nana", "Nancy",
  "Naomie", "Naseeruddin", "Natalia", "Natalie", "Natascha",
  "Nate", "Nathalie", "Nathan", "Nathaniel", "Naturi",
  "Naughton", "Nazanin", "NeNe", "Neal", "Neil",
  "Neill", "Nelly", "Nesbitt", "Nettles", "Neve",
  "Newberry", "Newton", "Niall", "Nichelle", "Nicholas",
  "Nichols", "Nicholson", "Nick", "Nicki", "Nicola",
  "Nicole", "Nighy", "Nikolaj", "Nina", "Niney",
  "Noah", "Noble", "Noel", "Nolasco", "Noomi",
  "Noriega", "Norman", "Novak", "Ochoa", "Octavia",
  "Odette", "Odeya", "Oh", "Okonedo", "Olga",
  "Oliver", "Olivia", "Om", "Omari", "Oona",
  "Ora", "Oscar", "Osmond", "Oswalt", "Owen",
  "Oyelowo", "P.", "Pablo", "Paddy", "Padma",
  "Paget", "Paisley", "Palicki", "Palin", "Palladio",
  "Palminteri", "Paloma", "Pamela", "Panabaker", "Paolo",
  "Parineeti", "Parker", "Parrish", "Pastore", "Pat",
  "Patinkin", "Patrick", "Patridge", "Pattinson", "Patton",
  "Paul", "Paulson", "Pearce", "Peck", "Pegg",
  "Peltz", "Penikett", "Penn", "Pennington", "Perlman",
  "Perry", "Pertwee", "Pete", "Peter", "Peterman",
  "Peters", "Peterson", "Pettyfer", "Philip", "Phillippe",
  "Phillips", "Phyllis", "Picardo", "Pierce", "Pierfrancesco",
  "Pierre", "Pieterse", "Pill", "Pine", "Pino",
  "Pinto", "Piven", "Placido", "Platt", "Pleshette",
  "Poehler", "Pollack", "Pooja", "Poppy", "Porter",
  "Portman", "Potter", "Potts", "Poulter", "Pounder",
  "Powers", "Preity", "Prescott", "Prince", "Pullman",
  "Punch", "Purefoy", "Puri", "Pyle", "Quentin",
  "Quinto", "R.", "RZA", "Rachel", "Rade",
  "Radnor", "Rahman", "Raimi", "Raini", "Rainn",
  "Raisa", "Rajskub", "Ralf", "Ralph", "Ramamurthy",
  "Rami", "Ramirez", "Ramsay", "Ranaut", "Ranbir",
  "Rancic", "Randeep", "Randolph", "Rannells", "Raoul",
  "Rapace", "Rapp", "Rauch", "Raver", "Ray",
  "Raymond", "Raymund", "Reba", "Rebecca", "Rebel",
  "Reece", "Reedus", "Reese", "Regan", "Reggie",
  "Regina", "Reid", "Reilly", "Reiner", "Rene",
  "Renner", "Reno", "Retta", "Reubens", "Reynolds",
  "Reynor", "Rhea", "Rheon", "Rhimes", "Rhys",
  "Rhys-Davies", "Ribisi", "Ricci", "Richard", "Richards",
  "Richardson", "Richie", "Richter", "Rickles", "Rickman",
  "Ricky", "Rico", "Riggle", "Rik", "Riley",
  "Ringwald", "Rinna", "Riseborough", "Rita", "Ritchie",
  "Rivers", "Riz", "Roach", "Rob", "Robbie",
  "Robert", "Roberts", "Robertson", "Robinson", "Rock",
  "Roday", "Rodriguez", "Rogen", "Roger", "Rogers",
  "Rohrwacher", "Roma", "Romeo", "Ron", "Ronda",
  "Rooney", "Rory", "Rose", "Roshon", "Rosie",
  "Ross", "Rossi", "Roth", "Rousey", "Rowan",
  "Rowell", "Roy", "Ruby", "Rudolph", "Rue",
  "Ruffalo", "Rupp", "Rush", "Russell", "Ruta",
  "Rutger", "Ruth", "Rutherford", "Ryan", "Rycroft",
  "S.", "Sackhoff", "Sagal", "Saldana", "Sally",
  "Salonga", "Sam", "Samantha", "Samberg", "Samira",
  "Sammi", "Sammy", "Samuel", "Sanaa", "Sandra",
  "Sands", "Sanon", "Santiago", "Sara", "Sarah",
  "Sasha", "Saum", "Saunders", "Schaal", "Schaech",
  "Scheer", "Scherzinger", "Schneider", "Schreiber", "Schumer",
  "Schwimmer", "Scodelario", "Scorsone", "Scorupco", "Scott",
  "Sean", "Seann", "Sebastian", "Seda", "Segel",
  "Sela", "Selleck", "Selma", "Sendhil", "Senta",
  "Serbedzija", "Serinda", "Seth", "Seyfried", "Shada",
  "Shah", "Shahid", "Shailene", "Shakur", "Shane",
  "Shannen", "Shannon", "Sharapova", "Sharlto", "Sharma",
  "Sharni", "Sharon", "Shaw", "Shay", "Shaye",
  "Shayk", "Sheehan", "Sheindlin", "Shelton", "Shemar",
  "Shepherd", "Sheppard", "Sherawat", "Sheridan", "Sherri",
  "Sheryl", "Shiloh", "Shipka", "Shiri", "Shirley",
  "Shohreh", "Shonda", "Shraddha", "Shum", "Siddig",
  "Sidharth", "Sidse", "Sienna", "Sigler", "Silas",
  "Simm", "Simmons", "Simon", "Simone", "Sisto",
  "Sivan", "Skylar", "Slate", "Slattery", "Smart",
  "Smit-McPhee", "Smith", "Smokey", "Snipes", "Snyder",
  "Socha", "Soderbergh", "Sofia", "Sohn", "Solo",
  "Somerhalder", "Sonam", "Song", "Sonja", "Sophie",
  "Sorrentino", "Sorvino", "Souza", "Spall", "Sparks",
  "Spearritt", "Speedman", "Spencer", "Stacy", "Stafford",
  "Stamp", "Stan", "Stana", "Stanley", "Staunton",
  "Stefanie", "Steinem", "Steinfeld", "Stephen", "Stephens",
  "Sterling", "Steve", "Steven", "Stevens", "Stevie",
  "Stewart", "Stiles", "Stiller", "Stoll", "Stone",
  "Strong", "Sturgess", "Suchet", "Sulkin", "Sung",
  "Sunny", "Suraj", "Susan", "Susanne", "Sutherland",
  "Suvari", "Suzanne", "Swan", "Swank", "Swayze",
  "Swift", "Swinton", "Swit", "Sydney", "Sykes",
  "T.", "T.J.", "Tabu", "Tahj", "Tahmoh",
  "Tamala", "Tamara", "Tambor", "Tamer", "Tamera",
  "Tamsin", "Tanya", "Taraji", "Tarantino", "Taryn",
  "Tate", "Tatiana", "Tatyana", "Tavi", "Taylor",
  "Teala", "Ted", "Teller", "Tena", "Tennant",
  "Terence", "Teri", "Terrence", "Terry", "Thandie",
  "Theler", "Theo", "Theroux", "Thieriot", "Thirlby",
  "Thomas", "Thompson", "Thorne", "Thune", "Tilda",
  "Tim", "Timothy", "Tinsel", "Titus", "Toby",
  "Todd", "Tom", "Tomlinson", "Tony", "Tosh",
  "Tovey", "Tracey", "Tracy", "Trainor", "Trammell",
  "Treadaway", "Trebek", "Trevino", "Tristan", "Troye",
  "Tucci", "Tudyk", "Tunie", "Tupac", "Turner",
  "Turturro", "Tveit", "Ty", "Tye", "Tyler",
  "Tyson", "Ulliel", "Underwood", "Usher", "Uzo",
  "Van", "Vanessa", "Vangsness", "Vanilla", "Varma",
  "Vaughn", "Verbeek", "Veronica", "Vicky", "Victoria",
  "Vidya", "Viggo", "Vince", "Vincent", "Vinson",
  "Violante", "Virginia", "Visitor", "Visnjic", "Voegele",
  "Walker", "Wallach", "Walsh", "Walter", "Walters",
  "Walton", "Wanda", "Ward", "Wareing", "Warner",
  "Wasikowska", "Waterston", "Watling", "Wayans", "Wayne",
  "Weatherly", "Webb", "Webber", "Weir", "Welliver",
  "Wells", "Wendell", "Wenders", "Wendi", "Wendy",
  "Wenham", "Wentworth", "Wentz", "Wes", "Wesley",
  "West", "Westwick", "Whedon", "Whelchel", "Whishaw",
  "Whitaker", "White", "Whitehall", "Whitley", "Whitney",
  "Wiig", "Wilde", "Wilds", "Wiley", "Wilkinson",
  "Will", "Willem", "William", "Williams", "Williams-Paisley",
  "Williamson", "Wilson", "Wim", "Winnick", "Winstead",
  "Winstone", "Witney", "Wonder", "Wong", "Wood",
  "Woodard", "Woodley", "Woods", "Woody", "Worthington",
  "Wright", "Xavier", "Yeardley", "Yelchin", "Yen",
  "Yeun", "Yoo", "Young", "Yun", "Yung",
  "Yvette", "Zac", "Zach", "Zachary", "Zack",
  "Zayas", "Zea", "Zimmer", "Zinta", "Zoe",
  "Zolciak-Biermann", "Zulay", "de", "deGrasse", "the",
  "" 
  ]

#arr = []
#arr = [chr(i) for i in range(97, 97 + 26)]

words = '["{}", "[unk]"]'.format(' '.join(arr))


if __name__ == "__main__":
  print(words)
  from vosk import Model, KaldiRecognizer
  import sys
  import os
  import wave

  model = Model("vosk-model-small-en-us-0.15")
  rec = KaldiRecognizer(model, 48000, words)
