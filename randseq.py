import argparse
import random

__version__ = "1.10"


help_message = f"""\033[1m\033[7mRandseq {__version__}\033[0m: 
A random biological sequence (DNA, RNA or protein) generator.

Author     : CDoo
Last update: 2024-10-07
Environment: Python 3.10.14
Version    : {__version__}

\033[1m\033[4mUSAGE\033[0m:
   randseq [-l [LENGTH...]] [-n NUMBER] [-t [TYPE] [--GC [GC]]] [-d [DELIMITER]] [-f] [-T] [-U -L / -3] [-h]

\033[1m\033[4mOPTIONS\033[0m:
  -l, --length=[LENGTH...]          Length of sequences to be generated. Accepts 1 or 2 positive integers.
                                    Were 2 integers given, the length of each sequence will be randomly chosen between them.

  -n, --number=NUMBER               Number of sequences to be generated. 1 by default.

  -t, --type=TYPE                   Molecular type of sequences to be generated.
                                    Option: DNA/dna/d/D/NA/na, RNA/rna/r/R or protein/p. "protein" by default.

  --GC=GC                           GC percent of the nucleic acid sequence. Available only when --type is in DNA or RNA.
                                    Must be within [0, 1].
                                    Note: Due to the randomness of sequence generation, the GC percent of generated sequences
                                    is very likely to be close but not exactly equal to the GC percent value provided by user.

  The exact percentage of each nucleotide is randomly calculated, according to the following 4 arguments:
    --gwidth=GWIDTH
    --gmedian=GMEDIAN
    --awidth=AWIDTH
    --amedian=AMEDIAN
  These arguments define the percentage of single nucleotide within range [a, b]×GC% (or AT%), where
  a = (2 * median - width)/2;
  b = a + width.
  Median and width are the middle point & width of the range, repectively.

  -d, --delimiter=DELIMITER         A symbol that seperates amino acid or nucleotide residues. NULL by default.

  -T, --termini                     Whether to add terminal signature at both ends of the sequence.
                                    If --type is nucleic acid (RNA/DNA), will be "5'-" and "-3'";
                                    If --type is protein, will be 'NH3-' and '-COOH'.

  -f, --fasta                       Whether to output with fasta format, which is the following:
                                    > \033[3mRandom [--type] sequence\033[0m
                                    ...SEQUENCE...

  The following options are mutually exclusive:
    -U, --upper                       Whether to use upper-case letter code.
    -L, --lower                       Whether to use lower-case letter code.
    -3, --three-letter                Whether to use three-letter code for protein sequences.
                                      Available only when --type is "protein".

\033[1m\033[4mExamples\033[0m:
randseq --length=300 --number=10  # Randomly generate 10 sequences with 300 amino acid residues
randseq -l 50 1000 -f > seq.fa    # Generate a 50 - 1000-residue long sequence and write into the file seq.fa
randseq -d - -3 -T                # Generate a sequence that has 'NH3-Xaa1-Xaa2-Xaa3...-COOH' format
randseq --type=DNA --GC=0.3 -n 5  # Generate 5 DNA sequences whose GC percent is 30%."""


def _base_alloc(GC, gwidth, gmedian, awidth, amedian):
    ag, aa = (2*gmedian - gwidth)/2, (2*amedian - awidth)/2
    bg, ba = ag + gwidth, aa + awidth
    assert (0 <= GC <= 1) and (bg >= ag) and (ag >= 0) and (bg <= 1)
    assert (ba >= aa) and (aa >= 0) and (ba <= 1)
    x, y = random.random(), random.random()
    G = (gwidth * x + ag)*GC
    C = GC - G
    AT = 1 - GC
    A  = (awidth * y + aa)*AT
    T  = AT - A
    return ([A, G, C, T], [ag, bg], [aa, ba])


def _parse_gc(parser):
    parser.add_argument("--GC",    type=float, default=0.5)
    parser.add_argument("--gwidth",  type=float, default=0.0)
    parser.add_argument("--gmedian", type=float, default=0.5)
    parser.add_argument("--awidth",  type=float, default=0.0)
    parser.add_argument("--amedian", type=float, default=0.5)
    na_params = parser.parse_known_args()[0]
    return _base_alloc(na_params.GC, 
                       na_params.gwidth, na_params.gmedian,
                       na_params.awidth, na_params.amedian)


def _termini(type):
    return ['NH3-', '-COOH'] if type == 'protein' else ["5'-", "-3'"]


def main():
    # --------------------------------------
    # Define the bulk of the parser.
    # --------------------------------------
    psr = argparse.ArgumentParser(
        prog     = f"Randseq {__version__}",
        add_help = False          )

    # --------------------------------------
    # Define arguments.
    # --------------------------------------
    # Firstly: Parse the molecular type of sequences.
    # Some arguments are available only when specific molecular type is provided.
    # And some arguments differ in meanings under distinct circumstances.
    psr.add_argument('-t', "--type", default='protein')
    chem = (psr.parse_known_args()[0]).type

    # Secondly: Parse the arguments related to molecular type and calculate the weights.
    # The specification of amino acid weights is currently not available.
    # There is a wide range of typing, but in this program, 
    # they will be recognized, stored, and treated as formal chemical identities. 
    if chem in ('p', 'P', 'protein'):
        chem, letter = 'protein', 'ACDEFGHIKLMNPQRSTVWY'
        weights  = [0.05]*20
    elif chem in ('d', 'D', 'dna', 'DNA', 'NA', 'na'):
        chem, letter = 'DNA', 'AGCT'
        weights, grange, arange = _parse_gc(psr)
    elif chem in ('r', 'R', 'rna', 'RNA'):
        chem, letter = 'RNA', 'AGCU'
        weights, grange, arange = _parse_gc(psr)
    else:
        raise ValueError(f'Invalid type of sequences: {chem}')

    # Basic information needed to generate random sequences.
    # Option [--length]: If a number range [n, m] is given, 
    #                    then the lengths of generated sequences 
    #                    will be randomly chosen between n and m.
    psr.add_argument('-l', "--length", type=int,  nargs="*", default=[50, 1000])
    psr.add_argument('-n', "--number", type=int,             default=1)
    
    # A bit of format support.
    format_selector = psr.add_mutually_exclusive_group()
    format_selector.add_argument('-f', "--fasta", const=f"Random {chem} sequence", action='store_const')
    
    sformater = format_selector.add_argument_group()
    sformater.add_argument('-d', "--delimiter", default="")
    sformater.add_argument("-T", "--termini", action='store_true')
    
    # Upper-case and lower-case option must not be simultaneously true.
    casefolder = sformater.add_mutually_exclusive_group()
    casefolder.add_argument("-U", "--upper-case"  , dest='upper', action='store_true')
    casefolder.add_argument("-L", "--lower-case"  , dest='lower', action='store_true')
    if chem == 'protein':
        casefolder.add_argument("-3", "--three-letter", dest='three', action='store_true')

    # Message support for debugging and user convenience.
    psr.add_argument('-h', '--help', action='store_true')
    psr.add_argument('-N', '--note', action='store_true')


    # --------------------------------------
    # Execute argument parsing and sequence generation.
    # --------------------------------------
    args = psr.parse_args()
    
    # Print help message and exit upon occurence of --help/-h option.
    if args.help:
        print(help_message)
        psr.exit(1)
    
    # Calculate factors used to construct random sequences.
    termini = _termini(chem) if args.termini else ["", ""]
    letter  = letter.lower() if args.lower   else letter.upper()

    try:
    # If the sequences are to be output in fasta format, 
    # there must not be any additional format modifications.
        if args.fasta and (args.delimiter or args.termini or args.lower or args.three):
            raise ValueError('Fasta format must not coexist with format modifications')
    # Three-letter code is only available for protein sequences.
        if args.three:
            if chem == 'protein':
                letter = ['Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His', 'Ile', 'Lys', 'Leu',
                          'Met', 'Asn', 'Pro', 'Gln', 'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr']
            else:
                raise ValueError('Nucleic acid type has no three-letter code')
    except AttributeError: pass
    
    # Print user-defined arguments and factors calculated, after which sequences are generated.
    if args.note:
        kwargs = args._get_kwargs()
        space = len(max(kwargs, key=lambda x: len(x[0]))[0]) 
        if chem in ('DNA', 'RNA'):
            g_percent_range = [grange[0]*args.GC, grange[1]*args.GC]
            a_percent_range = [arange[0]*(1-args.GC), arange[1]*(1-args.GC)]
            note_message="\033[1mArguments received\033[0m:\n" + \
                         "\n".join([("{:%s}: {}"%space).format(attr, value) for attr, value in kwargs]) + \
                         "\n\n" + \
                         "\033[1mFactors calculated or deduced from arguments\033[0m:\n" + \
                         f"    Molecular type: {chem}\n" + \
                         f"              Code: {letter}\n" + \
                         f"           Weights: {weights}\n" + \
                         f"        G interval: {grange}\n" + \
                         f"   G total percent: {g_percent_range}\n" + \
                         f"        A interval: {arange}\n" + \
                         f"   A total percent: {a_percent_range}\n" + \
                         f"      Termini used: {termini}\n" + \
                         "The factors listed above will be used to generate random sequences.\n"
        else:
            note_message="\033[1mArguments received\033[0m:\n" + \
                         "\n".join([("{:%s}: {}"%space).format(attr, value) for attr, value in kwargs]) + \
                         "\n\n" + \
                         "\033[1mFactors calculated or deduced from arguments\033[0m:\n" + \
                         f"    Molecular type: {chem}\n" + \
                         f"              Code: {letter if not args.three else ' '.join(letter)}\n" + \
                          "           Weights: All equal\n" + \
                         f"      Termini used: {termini}\n" + \
                         "The factors listed above will be used to generate random sequences.\n"
        print(note_message)

    if len(args.length) == 1:
        length = args.length[0]
        for i in range(args.number):
            if args.fasta:
                print(">" + args.fasta + " " + str(i + 1))
            # construct and output the final sequence
            seq = termini[0] + \
                  args.delimiter.join(
                      random.choices(
                          population=letter,
                          weights=weights,
                          k=length   )) + \
                  termini[1]
            print(seq)

    elif len(args.length) == 2:
        start, end = min(args.length), max(args.length)
        for i in range(args.number):
            if args.fasta:
                print(">" + args.fasta + " " + str(i + 1))
            seq = termini[0] + \
                  args.delimiter.join(
                      random.choices(
                          population=letter,
                          weights=weights,
                          k=random.randint(start, end) # Randomly choose the length in the range
                                     )) + \
                  termini[1]
            print(seq)
    else:
        raise ValueError('Unknown length range: ' + (f'{L for L in args.length}'))


if __name__ == "__main__":
    main()



