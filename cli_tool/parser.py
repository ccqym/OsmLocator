import argparse

def getParser():
    #initialize parser
    parser = argparse.ArgumentParser(
            prog = 'OsmLocator',
            description = 'A tool for locating overlapping scatter marks. If you would like to use this package, please cite the paper: Yuming Qiu, Aleksandra Pizurica, Qi Ming and Nicolas Nadisic, OsmLocator: locating overlapping scatter marks by simulated annealing on clustering-based re-visualization, 2023',
            epilog = 'OsmLocator v1.0, https://github.com/ccqym/OsmLocator')

    parser.add_argument('input', metavar='input', type=str, help='specify the path of scatter image file to be handled.')

    #optional arguments
    parser.add_argument('-o', '--output', default=None, type=str, help = 'specify the path of Json formatted output file.')
    parser.add_argument('-O', '--visualize-outfile', default=None, type=str, help = 'specify the path of visualized image file.')
    parser.add_argument('-a', '--alpha', default=1.1, type=float, help = 'alpha .')
    parser.add_argument('-b', '--beta', default=1, type=float, help = 'beta.')
    parser.add_argument('-g', '--gt', default=None, type=str, help = 'gt file path.')
    parser.add_argument('-C', '--space-setting-factor', default=60, type=int, help = 'a constant.')
    parser.add_argument('--blur-kernel-size', default=7, type=int, help = 'kernel size for Gaussian blur.')
    parser.add_argument('-f', '--impact-of-recognized-marks', default=0.3, type=float, help = 'factor of search space adjust.')
    parser.add_argument('--is-auto-search-space', default=True, action=argparse.BooleanOptionalAction, help = 'auto detect search space.')
    parser.add_argument('--gamma-markov', default=1.5, type=float, help = 'coeficient of length of Markov chain.')
    parser.add_argument('--gamma-stop', default=1.5, type=float, help = 'coefficient of stop criteria of simulated annealing.')
    parser.add_argument('--out-point-size', default=12, type=int, help = 'output data point size.')
    parser.add_argument('--out-point-color', default='k', type=str, help = 'output data point color.')
    parser.add_argument('--supportable-markers', default='o,s,D', type=str, help = 'supportable markers.')
    parser.add_argument('--is-remove-text-and-axis', default=False, action=argparse.BooleanOptionalAction, help = 'is remove text and axis?')
    parser.add_argument('--remove-kernel-size', default=11, type=int, help = 'kernel size for removing text and axises.')
    parser.add_argument('--is-hsv-s-channel', default=False, action=argparse.BooleanOptionalAction, help = 'auto detect search space.')
    parser.add_argument('--binarize-method', default='ostu', type=str, help = 'the method to binarize. ostu or threshold')
    parser.add_argument('--binarize-threshold', default=180, type=int, help = 'the threshold value for binarization.')
    parser.add_argument('-V', '--verbose', default=False, action=argparse.BooleanOptionalAction, help = 'show infomation.')

    #read arguments from cli
    args = parser.parse_args()
    assert(args != None)
    print(args)
    return args

def convertToSettings(args):
    ss = {}
    ss['supportable_markers'] = args.supportable_markers
    ss['space_setting_factor'] = args.space_setting_factor
    ss['is_auto_search_space'] = args.is_auto_search_space
    ss['impact_of_recognized_marks'] = args.impact_of_recognized_marks
    ss['gamma_markov'] = args.gamma_markov
    ss['gamma_stop'] = args.gamma_stop
    ss['alpha'] = args.alpha
    ss['beta'] = args.beta
    ss['verbose'] = args.verbose
    return ss

