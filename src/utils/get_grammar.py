from pcfg import PCFG
import numpy as np
from definitions import ROOT_DIR


def read_grammar_file(args):
    path = ROOT_DIR / args.data_path
    with open(path / 'production_rules.txt') as file:
        content = file.read()
    grammar = PCFG.fromstring(content)
    add_prior(grammar=grammar, args=args)
    return grammar


def add_prior(grammar, args):
    productions = grammar._productions  # noqa W0212
    non_terminals = list(grammar._lhs_index.keys())  # noqa W0212
    prior_dict = {}
    for non_terminal in non_terminals:
        if args.prior_source == 'grammar':
            non_terminal_prior = get_grammar_prior(
                non_terminal=non_terminal,
                productions=productions
            )
        elif args.prior_source == 'uniform':
            non_terminal_prior = get_uniform_prior(
                non_terminal=non_terminal,
                productions=productions
            )
        else:
            raise RuntimeError(f"Prior '{args.prior_source}' not supported.")
        prior_dict[str(non_terminal)] = non_terminal_prior
    grammar.prior_dict = prior_dict


def get_grammar_prior(non_terminal, productions):
    non_terminal_prior = []
    for production in productions:
        if production._lhs == non_terminal:  # noqa W0212
            non_terminal_prior.append(production._ProbabilisticMixIn__prob)  # noqa W0212
        else:
            non_terminal_prior.append(0)
    return non_terminal_prior


def get_uniform_prior(non_terminal, productions):
    non_terminal_prior = []
    for production in productions:
        if production._lhs == non_terminal:  # noqa W0212
            non_terminal_prior.append(1)
        else:
            non_terminal_prior.append(0)
    non_terminal_prior = np.array(non_terminal_prior)
    non_terminal_prior = non_terminal_prior / np.sum(non_terminal_prior)
    return non_terminal_prior
