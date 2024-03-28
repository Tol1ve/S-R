import types
import warnings
import argparse
import logging
from typing import cast


NOT_DOC = True


def doc_mode():
    global NOT_DOC
    NOT_DOC = False


def not_doc():
    return NOT_DOC


class RST:
    def __init__(self, rst) -> None:
        self.rst = rst

    def __str__(self) -> str:
        return rst2txt(self.rst)


def rst(source: str):
    if not NOT_DOC:
        return source
    else:
        return RST(source)


def rst2txt(source: str) -> str:
    """
    adapted from https://stackoverflow.com/questions/57119361/convert-restructuredtext-to-plain-text-programmatically-in-python
    """
    try:
        import docutils.nodes
        import docutils.parsers.rst
        import docutils.utils
        import sphinx.writers.text
        import sphinx.builders.text
        import sphinx.util.osutil
        from sphinx.application import Sphinx

        # parser rst
        parser = docutils.parsers.rst.Parser()
        components = (docutils.parsers.rst.Parser,)
        settings = docutils.frontend.OptionParser(
            components=components
        ).get_default_values()
        document = docutils.utils.new_document("<rst-doc>", settings=settings)
        parser.parse(source, document)

        # sphinx
        _app = types.SimpleNamespace(
            srcdir=None,
            confdir=None,
            outdir=None,
            doctreedir="/",
            events=None,
            config=types.SimpleNamespace(
                text_newlines="native",
                text_sectionchars="=",
                text_add_secnumbers=False,
                text_secnumber_suffix=".",
            ),
            tags=set(),
            registry=types.SimpleNamespace(
                create_translator=lambda self, something, new_builder: sphinx.writers.text.TextTranslator(
                    document, new_builder
                )
            ),
        )
        app = cast(Sphinx, _app)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builder = sphinx.builders.text.TextBuilder(app)
        translator = sphinx.writers.text.TextTranslator(document, builder)
        document.walkabout(translator)
        return str(translator.body)
    except Exception as e:
        logging.warning("Got the following error during rst conversion: %s", e)
        return source


def show_link(text: str, link: str) -> str:
    if NOT_DOC:
        return link
    else:
        return f"`{text} <{link}>`_"


def get_subparsers_action(
    parser: argparse.ArgumentParser,
) -> argparse._SubParsersAction:
    subparsers_action: argparse._SubParsersAction
    for action_group in parser._action_groups:
        for action in action_group._group_actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
    return subparsers_action


def prepare_parser_for_sphinx(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    # find _SubParsersAction
    subparsersaction = get_subparsers_action(parser)

    # update epilog
    epilog = (
        ".. list-table:: \n"
        "   :widths: 30 70 \n"
        "   :header-rows: 1 \n\n"
        "   * - Command \n"
        "     - Description\n"
    )
    for subaction in subparsersaction._get_subactions():
        # epilog += f"\n:doc:`{subaction.dest}`\n    {subaction.help}\n"
        epilog += f"   * - :doc:`{subaction.dest}`\n" f"     - {subaction.help}\n"
    parser.epilog = epilog

    for name, subparser in subparsersaction._name_parser_map.items():
        for action_group in subparser._action_groups:
            for action in action_group._group_actions:
                if isinstance(action, argparse._HelpAction):
                    continue
                help = ""
                if action.choices is not None:
                    choices = [f"`{c}`" for c in action.choices]
                    help += f"**Possible choices**: {', '.join(choices)}\n\n"
                    action.choices = None
                elif action.type is not None:
                    t = getattr(action.type, "__name__", str(action.type))
                    if action.nargs == "+":
                        t = f"{t} [{t} ...]"
                    elif action.nargs in [None, 0, 1]:
                        pass
                    else:
                        raise NotImplementedError(t, action.nargs)
                    help += f"**Type**: `{t}`\n\n"
                    # action.nargs
                if action.default is not None and not isinstance(
                    action, argparse._StoreConstAction
                ):
                    help += f"**Default**: `{action.default}`\n\n"
                if action.help is None:
                    action.help = help
                else:
                    action.help = help + action.help
    return parser
