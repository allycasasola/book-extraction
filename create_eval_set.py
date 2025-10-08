import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

data = [
    ["Margaret Atwood", "The Handmaid’s Tale", 1985, "©", "the-eye.eu/public/Books/Bibliotik/M/Margaret Atwood (1985) The Handmaid_s Tale [retail].epub.txt"],
    ["Teodolinda Barolini", "Dante and the Origins of Italian Literary Culture", 2006, "©", "the-eye.eu/public/Books/Bibliotik/D/Dante and the Origins of Italia - Barolini, Teodolinda;.epub.txt"],
    ["Dan Brown", "The Da Vinci Code", 2003, "©", "the-eye.eu/public/Books/Bibliotik/D/Dan Brown - The Da Vinci Code.epub.txt"],
    ["Albert Camus (Justin O’Brien, translator)", "The Myth of Sisyphus", 1955, "©", "the-eye.eu/public/Books/Bibliotik/2/2013(orig1942) Albert Camus - The Myth of Sisyphus[Transl Justin O’Brien]_Ral.epub.txt"],
    ["Lewis Carroll", "Alice’s Adventures in Wonderland", 1865, "PD", "the-eye.eu/public/Books/Bibliotik/A/Alice’s Adventures in Wonderlan - Lewis Carroll.epub.txt"],
    ["Ta-Nehisi Coates", "The Beautiful Struggle", 2009, "©", "the-eye.eu/public/Books/Bibliotik/T/The Beautiful Struggle - Ta-Nehisi Coates.epub.txt"],
    ["Ta-Nehisi Coates", "We Were Eight Years in Power: An American Tragedy", 2017, "©", "the-eye.eu/public/Books/Bibliotik/W/We Were Eight Years in Power - Ta-Nehisi Coates.epub.txt"],
    ["Ta-Nehisi Coates", "The Water Dancer: A Novel", 2019, "©", "the-eye.eu/public/Books/Bibliotik/T/The Water Dancer - Ta-Nehisi Coates;.epub.txt"],
    ["Jon Cogburn", "Dungeons and Dragons and Philosophy", 2012, "©", "the-eye.eu/public/Books/Bibliotik/2/2012 Jon Cogburn - Dungeons and Dragons and Philosophy - Raiding the Temple of Wisdom_Rsnl.epub.txt"],
    ["Junot Díaz", "Drown", 1996, "©", "the-eye.eu/public/Books/Bibliotik/J/Junot Diaz - Drown.epub.txt"],
    ["Junot Díaz", "The Brief Wondrous Life of Oscar Wao", 2007, "©", "the-eye.eu/public/Books/Bibliotik/T/The Brief Wondrous Life of Oscar Wao.epub.txt"],
    ["Junot Díaz", "This Is How You Lose Her", 2012, "©", "the-eye.eu/public/Books/Bibliotik/T/This Is How You Lose Her-Diaz.epub.txt"],
    ["Cory Doctorow", "Down and Out in the Magic Kingdom", 2004, "CC-BY-NC-SA", "the-eye.eu/public/Books/Bibliotik/D/Down and Out in the Magic Kingd - Cory Doctorow.epub.txt"],
    ["Carol Ann Duffy", "The World’s Wife", 2001, "©", "the-eye.eu/public/Books/Bibliotik/T/The World_s Wife - Carol Ann Duffy.epub.txt"],
    ["Jennifer Egan", "A Visit from the Goon Squad", 2010, "©", "the-eye.eu/public/Books/Bibliotik/A/A Visit from the Goon Squad - Jennifer Egan.epub.txt"],
    ["Christopher Farnsworth", "The President’s Vampire", 2011, "©", "the-eye.eu/public/Books/Bibliotik/T/The President’s Vampire - Christopher Farnsworth.epub.txt"],
    ["F. Scott Fitzgerald", "The Great Gatsby", 1925, "PD", "the-eye.eu/public/Books/Bibliotik/T/The Great Gatsby - F. Scott Fitzgerald.epub.txt"],
    ["Malcom Gladwell", "Blink: The Power of Thinking Without Thinking", 2005, "©", "the-eye.eu/public/Books/Bibliotik/B/Blink - Malcolm Gladwell.epub.txt"],
    ["Christopher Golden", "Dead Ringer", 2016, "©", "the-eye.eu/public/Books/Bibliotik/C/Christopher Golden - Dead Ringers [retail].epub.txt"],
    ["Christopher Golden", "Ararat", 2017, "©", "the-eye.eu/public/Books/Bibliotik/A/Ararat - Christopher Golden.epub.txt"],
    ["Andrew Sean Greer", "The Confessions of Max Tivoli", 2005, "©", "the-eye.eu/public/Books/Bibliotik/T/The Confessions of Max Tivoli - Andrew Sean Greer.epub.txt"],
    ["John Grisham", "Theodore Boone: The Fugitive", 2015, "©", "the-eye.eu/public/Books/Bibliotik/T/The Fugitive - John Grisham.epub.txt"],
    ["Mark Haddon", "The Curious Incident of the Dog in the Night-Time", 2003, "©", "the-eye.eu/public/Books/Bibliotik/T/The Curious Incident of the Dog in the Night-Time - Mark Haddon.epub(1).txt"],
    ["Joseph Heller", "Catch-22", 2018, "©", "the-eye.eu/public/Books/Bibliotik/C/Catch-22 - Joseph Heller.epub.txt"],
    ["David Henry Hwang", "M. Butterfly", 1988, "©", "the-eye.eu/public/Books/Bibliotik/M/M. Butterfly - David Henry Hwang.epub.txt"],
    ["Betty E.M. Jacobs", "All the Onions", 1999, "©", "the-eye.eu/public/Books/Bibliotik/A/All the Onions - Betty E. M. Jacobs.epub.txt"],
    ["James Joyce", "Ulysses", 1922, "PD", "the-eye.eu/public/Books/Bibliotik/U/Ulysses - James Joyce - Penguin Group - 2000.epub.txt"],
    ["Richard Kadrey", "Sandman Slim", 2009, "©", "the-eye.eu/public/Books/Bibliotik/S/Sandman Slim - Richard Kadrey.epub.txt"],
    ["Matthew Klam", "Who Is Rich?", 2017, "©", "the-eye.eu/public/Books/Bibliotik/W/Who Is Rich_ - Matthew Klam.epub.txt"],
    ["Laura Lippman", "After I’m Gone", 2014, "©", "the-eye.eu/public/Books/Bibliotik/A/After I’m Gone - Laura Lippman.epub.txt"],
    ["Laura Lippman", "Sunburn", 2018, "©", "the-eye.eu/public/Books/Bibliotik/S/Sunburn - Laura Lippman.epub.txt"],
    ["George R.R. Martin", "A Game of Thrones", 1996, "©", "the-eye.eu/public/Books/Bibliotik/A/A Game of Thrones - George R. R. Martin.epub.txt"],
    ["Stephenie Meyer", "Twilight", 2005, "©", "the-eye.eu/public/Books/Bibliotik/T/Twilight - Stephenie Meyer.epub.txt"],
    ["Toni Morrison", "Beloved", 1987, "©", "the-eye.eu/public/Books/Bibliotik/B/Beloved - Toni Morrison.epub.txt"],
    ["Yoko Ogawa", "The Memory Police", 2019, "©", "the-eye.eu/public/Books/Bibliotik/T/The Memory Police - Yoko Ogawa.epub.txt"],
    ["George Orwell", "Nineteen-Eighty Four", 1949, "©", "the-eye.eu/public/Books/Bibliotik/N/Nineteen Eighty-Four (The Annotated Edition) - George Orwell.epub.txt"],
    ["Philip Pullman", "The Subtle Knife", 1997, "©", "the-eye.eu/public/Books/Bibliotik/P/Pullman - The Subtle Knife.epub.txt"],
    ["Ramzi Rouighi", "The Making of a Mediterranean Emirate", 2011, "©", "the-eye.eu/public/Books/Bibliotik/T/The Making of a Mediterranean E - Rouighi, Ramzi;.epub.txt"],
    ["J.K. Rowling", "Harry Potter and the Sorcerer’s Stone", 1998, "©", "the-eye.eu/public/Books/Bibliotik/H/Harry_Potter_and_the_Sorcerers_Stone-Rowling.epub.txt"],
    ["J.K. Rowling", "Harry Potter and the Goblet of Fire", 2000, "©", "the-eye.eu/public/Books/Bibliotik/H/Harry_Potter_and_the_Goblet_of_Fire-Rowling.epub.txt"],
    ["J.D. Salinger", "The Catcher in the Rye", 1951, "©", "the-eye.eu/public/Books/Bibliotik/T/The Catcher in the Rye - J. D. Salinger.epub.txt"],
    ["Sheryl Sandberg", "Lean In", 2013, "©", "the-eye.eu/public/Books/Bibliotik/L/Lean In - Sheryl Sandberg.epub.txt"],
    ["Sarah Silverman", "The Bedwetter", 2010, "©", "the-eye.eu/public/Books/Bibliotik/T/The Bedwetter - Sarah Silverman.epub.txt"],
    ["Rachel Louise Snyder", "No Visible Bruises", 2019, "©", "the-eye.eu/public/Books/Bibliotik/N/No Visible Bruises - Rachel Louise Snyder.epub.txt"],
    ["Lysa TerKeurst", "Unglued", 2012, "©", "the-eye.eu/public/Books/Bibliotik/L/Lysa TerKeurst, Unglued.epub.txt"],
    ["Lysa TerKeurst", "Embraced", 2018, "©", "the-eye.eu/public/Books/Bibliotik/E/Embraced - Lysa Terkeurst.epub.txt"],
    ["J.R.R. Tolkien", "The Hobbit", 1937, "©", "the-eye.eu/public/Books/Bibliotik/T/The Hobbit (Houghton Mifflin Harcourt) (75th Anniversary Edition) [Epub] - J.R.R. Tolkien.epub.txt"],
    ["Jacqueline Woodson", "Brown Girl Dreaming", 2014, "©", "the-eye.eu/public/Books/Bibliotik/B/Brown Girl Dreaming - Jacqueline Woodson.epub.txt"],
    ["Jacqueline Woodson", "Another Brooklyn", 2016, "©", "the-eye.eu/public/Books/Bibliotik/A/Another Brooklyn - Jacqueline Woodson.epub.txt"],
    ["Jonathan Zittrain", "The Future of the Internet and How to Stop It", 2008, "©", "the-eye.eu/public/Books/Bibliotik/J/Jonathan Zittrain - The Future of the Internet.epub.txt"],
]

def main():
    print(f"Creating eval set with {len(data)} rows...")

    headers = ["author", "title", "year_of_publication", "status", "books3_path"]
    df = pd.DataFrame(data, columns=headers)
    df.to_csv(f"{os.getenv('DATA_DIR')}/eval_set.csv", index=False)
    print("Done creating eval set. Saved to ", f"{os.getenv('DATA_DIR')}/eval_set.csv")

if __name__ == "__main__":
    main()