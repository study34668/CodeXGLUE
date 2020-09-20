const PREC = {
  COMMENT: -2,
  CURLY_BLOCK: 1,
  DO_BLOCK: -1,

  AND: -2,
  OR: -2,
  NOT: 5,
  DEFINED: 10,
  ALIAS: 11,
  ASSIGN: 15,
  RESCUE: 16,
  CONDITIONAL: 20,
  RANGE: 25,
  BOOLEAN_OR: 30,
  BOOLEAN_AND: 35,
  RELATIONAL: 40,
  COMPARISON: 45,
  BITWISE_OR: 50,
  BITWISE_AND: 55,
  CALL: 56,
  SHIFT: 60,
  ADDITIVE: 65,
  MULTIPLICATIVE: 70,
  UNARY_MINUS: 75,
  EXPONENTIAL: 80,
  COMPLEMENT: 85,
};

const IDENTIFIER_CHARS = /[^\x00-\x1F\s:;`"'@$#.,|^&<=>+\-*/\\%?!~()\[\]{}]*/;
const LOWER_ALPHA_CHAR = /[^\x00-\x1F\sA-Z0-9:;`"'@$#.,|^&<=>+\-*/\\%?!~()\[\]{}]/;
const ALPHA_CHAR = /[^\x00-\x1F\s0-9:;`"'@$#.,|^&<=>+\-*/\\%?!~()\[\]{}]/;

module.exports = grammar({
  name: 'ruby',

  externals: $ => [
    $._line_break,

    // Delimited literals
    $._simple_symbol,
    $._string_start,
    $._symbol_start,
    $._subshell_start,
    $._regex_start,
    $._string_array_start,
    $._symbol_array_start,
    $._heredoc_body_start,
    $._string_content,
    $._heredoc_content,
    $._string_end,
    $.heredoc_end,
    $.heredoc_beginning,

    // Tokens that require lookahead
    '/',
    $._block_ampersand,
    $._splat_star,
    $._unary_minus,
    $._binary_minus,
    $._binary_star,
    $._singleton_class_left_angle_left_langle,
    $._identifier_hash_key
  ],

  extras: $ => [
    $.comment,
    $.heredoc_body,
    /\s|\\\n/
  ],

  word: $ => $.identifier,

  supertypes: $ => [
    $._statement,
    $._arg,
    $._method_name,
    $._variable,
    $._primary,
    $._lhs,
  ],

  rules: {
    program: $ => seq(
      optional($._statements),
      optional(seq(
        '__END__',
        $._line_break,
        $.uninterpreted)
      )
    ),

    uninterpreted: $ => /(.|\s)*/,

    _statements: $ => choice(
      seq(
        repeat1(choice(
          seq($._statement, $._terminator),
          $.empty_statement
        )),
        optional($._statement)
      ),
      $._statement
    ),

    begin_block: $ => seq("BEGIN", "{", optional($._statements), "}"),
    end_block: $ => seq("END", "{", optional($._statements), "}"),

    _statement: $ => choice(
      $.undef,
      $.alias,
      $.if_modifier,
      $.unless_modifier,
      $.while_modifier,
      $.until_modifier,
      $.rescue_modifier,
      $.begin_block,
      $.end_block,
      $._expression
    ),

    method: $ => seq('def', $._method_rest),

    singleton_method: $ => seq(
      'def',
      seq(
        choice(
          field('object', $._variable),
          seq('(', field('object', $._arg), ')')
        ),
        choice('.', '::')
      ),
      $._method_rest
    ),

    _method_rest: $ => seq(
      field('name', $._method_name),
      choice(
        seq(
          field('parameters', alias($.parameters, $.method_parameters)),
          optional($._terminator)
        ),
        seq(
          optional(
            field('parameters', alias($.bare_parameters, $.method_parameters))
          ),
          $._terminator
        ),
      ),
      $._body_statement
    ),

    parameters: $ => seq(
      '(',
      commaSep($._formal_parameter),
      ')'
    ),

    bare_parameters: $ => seq(
      $._simple_formal_parameter,
      repeat(seq(',', $._formal_parameter))
    ),

    block_parameters: $ => seq(
      '|',
      seq(commaSep($._formal_parameter), optional(',')),
      optional(seq(';', sep1($.identifier, ','))), // Block shadow args e.g. {|; a, b| ...}
      '|'
    ),

    _formal_parameter: $ => choice(
      $._simple_formal_parameter,
      alias($.parameters, $.destructured_parameter)
    ),

    _simple_formal_parameter: $ => choice(
      $.identifier,
      $.splat_parameter,
      $.hash_splat_parameter,
      $.block_parameter,
      $.keyword_parameter,
      $.optional_parameter
    ),

    splat_parameter: $ => seq(
      '*',
      field('name', optional($.identifier))
    ),
    hash_splat_parameter: $ => seq(
      '**',
      field('name', optional($.identifier))
    ),
    block_parameter: $ => seq(
      '&',
      field('name', $.identifier)
    ),
    keyword_parameter: $ => prec.right(PREC.BITWISE_OR + 1, seq(
      field('name', $.identifier),
      token.immediate(':'),
      field('value', optional($._arg))
    )),
    optional_parameter: $ => prec(PREC.BITWISE_OR + 1, seq(
      field('name', $.identifier),
      '=',
      field('value', $._arg)
    )),

    class: $ => seq(
      'class',
      field('name', choice($.constant, $.scope_resolution)),
      optional($.superclass),
      $._terminator,
      $._body_statement
    ),

    superclass: $ => seq('<', $._arg),

    singleton_class: $ => seq(
      'class',
      alias($._singleton_class_left_angle_left_langle, '<<'),
      field('value', $._arg),
      $._terminator,
      $._body_statement
    ),

    module: $ => seq(
      'module',
      field('name', choice($.constant, $.scope_resolution)),
      choice(
        seq($._terminator, $._body_statement),
        'end'
      )
    ),

    return_command: $ => prec.left(seq('return', alias($.command_argument_list, $.argument_list))),
    yield_command: $ => prec.left(seq('yield', alias($.command_argument_list, $.argument_list))),
    break_command: $ => prec.left(seq('break', alias($.command_argument_list, $.argument_list))),
    next_command: $ => prec.left(seq('next', alias($.command_argument_list, $.argument_list))),
    return: $ => prec.left(seq('return', optional($.argument_list))),
    yield: $ => prec.left(seq('yield', optional($.argument_list))),
    break: $ => prec.left(seq('break', optional($.argument_list))),
    next: $ => prec.left(seq('next', optional($.argument_list))),
    redo: $ => prec.left(seq('redo', optional($.argument_list))),
    retry: $ => prec.left(seq('retry', optional($.argument_list))),

    if_modifier: $ => prec(PREC.RESCUE, seq(
      field('body', $._statement),
      'if',
      field('condition', $._expression)
    )),

    unless_modifier: $ => prec(PREC.RESCUE, seq(
      field('body', $._statement),
      'unless',
      field('condition', $._expression)
    )),

    while_modifier: $ => prec(PREC.RESCUE, seq(
      field('body', $._statement),
      'while',
      field('condition', $._expression)
    )),

    until_modifier: $ => prec(PREC.RESCUE, seq(
      field('body', $._statement),
      'until',
      field('condition', $._expression)
    )),

    rescue_modifier: $ => prec(PREC.RESCUE, seq(
      field('body', $._statement),
      'rescue',
      field('handler', $._expression)
    )),

    while: $ => seq(
      'while',
      field('condition', $._arg),
      field('body', $.do)
    ),

    until: $ => seq(
      'until',
      field('condition', $._arg),
      field('body', $.do)
    ),

    for: $ => seq(
      'for',
      commaSep1(field('pattern', choice(
        $._lhs,
        $.rest_assignment,
        $.destructured_left_assignment
      ))),
      field('value', $.in),
      field('body', $.do)
    ),

    in: $ => seq('in', $._arg),
    do: $ => seq(
      choice('do', $._terminator),
      optional($._statements),
      'end'
    ),

    case: $ => seq(
      'case',
      field('value', optional($._arg)),
      $._terminator,
      repeat(';'),
      repeat($.when),
      optional($.else),
      'end'
    ),

    when: $ => seq(
      'when',
      field('pattern', commaSep1($.pattern)),
      choice($._terminator, field('body', $.then))
    ),

    pattern: $ => choice($._arg, $.splat_argument),

    if: $ => seq(
      'if',
      field('condition', $._statement),
      choice($._terminator, field('consequence', $.then)),
      field('alternative', optional(choice($.else, $.elsif))),
      'end'
    ),

    unless: $ => seq(
      'unless',
      field('condition', $._statement),
      choice($._terminator, field('consequence', $.then)),
      field('alternative', optional(choice($.else, $.elsif))),
      'end'
    ),

    elsif: $ => seq(
      'elsif',
      field('condition', $._statement),
      choice($._terminator, field('consequence', $.then)),
      field('alternative', optional(choice($.else, $.elsif)))
    ),

    else: $ => seq(
      'else',
      field('condition', optional($._terminator)),
      optional($._statements)
    ),

    then: $ => choice(
      seq(
        $._terminator,
        $._statements
      ),
      seq(
        optional($._terminator),
        'then',
        optional($._statements)
      )
    ),

    begin: $ => seq('begin', optional($._terminator), $._body_statement),

    ensure: $ => seq('ensure', optional($._statements)),

    rescue: $ => seq(
      'rescue',
      field('exceptions', optional($.exceptions)),
      field('variable', optional($.exception_variable)),
      choice(
        $._terminator,
        field('body', $.then)
      )
    ),

    exceptions: $ => commaSep1(choice($._arg, $.splat_argument)),

    exception_variable: $ => seq('=>', $._lhs),

    _body_statement: $ => seq(
      optional($._statements),
      repeat(choice($.rescue, $.else, $.ensure)),
      'end'
    ),

    // Method calls without parentheses (aka "command calls") are only allowed
    // in certain positions, like the top-level of a statement, the condition
    // of a postfix control-flow operator like `if`, or as the value of a
    // control-flow statement like `return`. In many other places, they're not
    // allowed.
    //
    // Because of this distinction, a lot of rules have two variants: the
    // normal variant, which can appear anywhere that an expression is valid,
    // and the "command" varaint, which is only valid in a more limited set of
    // positions, because it can contain "command calls".
    //
    // The `_expression` rule can appear in relatively few places, but can
    // contain command calls. The `_arg` rule can appear in many more places,
    // but cannot contain command calls (unless they are wrapped in parens).
    // This naming convention is based on Ruby's standard grammar.
    _expression: $ => choice(
      alias($.command_binary, $.binary),
      alias($.command_assignment, $.assignment),
      alias($.command_operator_assignment, $.operator_assignment),
      alias($.command_call, $.method_call),
      alias($.command_call_with_block, $.method_call),
      prec.left(alias($.chained_command_call, $.call)),
      alias($.return_command, $.return),
      alias($.yield_command, $.yield),
      alias($.break_command, $.break),
      alias($.next_command, $.next),
      $._arg
    ),

    _arg: $ => choice(
      $._primary,
      $.assignment,
      $.operator_assignment,
      $.conditional,
      $.range,
      $.binary,
      $.unary
    ),

    _primary: $ => choice(
      $.parenthesized_statements,
      $._lhs,
      $.array,
      $.string_array,
      $.symbol_array,
      $.hash,
      $.subshell,
      $.symbol,
      $.integer,
      $.float,
      $.complex,
      $.rational,
      $.string,
      $.character,
      $.chained_string,
      $.regex,
      $.lambda,
      $.method,
      $.singleton_method,
      $.class,
      $.singleton_class,
      $.module,
      $.begin,
      $.while,
      $.until,
      $.if,
      $.unless,
      $.for,
      $.case,
      $.return,
      $.yield,
      $.break,
      $.next,
      $.redo,
      $.retry,
      alias($.parenthesized_unary, $.unary),
      alias($.unary_literal, $.unary),
      $.heredoc_beginning
    ),

    parenthesized_statements: $ => seq('(', optional($._statements), ')'),

    element_reference: $ => prec.left(1, seq(
      field('object', $._primary),
      token.immediate('['),
      optional($._argument_list_with_trailing_comma),
      ']'
    )),

    scope_resolution: $ => prec.left(1, seq(
      choice(
        '::',
        seq(field('scope', $._primary), token.immediate('::'))
      ),
      field('name', choice($.identifier, $.constant))
    )),

    call: $ => prec.left(PREC.CALL, seq(
      field('receiver', $._primary),
      choice('.', '&.'),
      field('method', choice($.identifier, $.operator, $.constant, $.argument_list))
    )),

    command_call: $ => seq(
      field('method', choice(
        $._variable,
        $.scope_resolution,
        $.call,
        alias($.chained_command_call, $.call)
      )),
      field('arguments', alias($.command_argument_list, $.argument_list))
    ),

    command_call_with_block: $ => {
      const receiver = field('method', choice($._variable, $.scope_resolution, $.call))
      const arguments = field('arguments', alias($.command_argument_list, $.argument_list))
      const block = field('block', $.block)
      const doBlock = field('block', $.do_block)
      return choice(
        seq(receiver, prec(PREC.CURLY_BLOCK, seq(arguments, block))),
        seq(receiver, prec(PREC.DO_BLOCK, seq(arguments, doBlock))),
      )
    },

    chained_command_call: $ => seq(
      field('receiver', alias($.command_call_with_block, $.method_call)),
      choice('.', '&.'),
      field('method', choice($.identifier, $.operator, $.constant, $.argument_list))
    ),

    method_call: $ => {
      const receiver = field('method', choice(
        $._variable,
        $.scope_resolution,
        $.call
      ))
      const arguments = field('arguments', $.argument_list)
      const block = field('block', $.block)
      const doBlock = field('block', $.do_block)
      return choice(
        seq(receiver, arguments),
        seq(receiver, prec(PREC.CURLY_BLOCK, seq(arguments, block))),
        seq(receiver, prec(PREC.DO_BLOCK, seq(arguments, doBlock))),
        prec(PREC.CURLY_BLOCK, seq(receiver, block)),
        prec(PREC.DO_BLOCK, seq(receiver, doBlock))
      )
    },

    command_argument_list: $ => choice(
      commaSep1($._argument),
      alias($.command_call, $.method_call),
      alias($.command_call_with_block, $.method_call)
    ),

    argument_list: $ => prec.right(seq(
      token.immediate('('),
      optional($._argument_list_with_trailing_comma),
      ')'
    )),

    _argument_list_with_trailing_comma: $ => prec.right(seq(
      commaSep1($._argument),
      optional(',')
    )),

    _argument: $ => choice(
      $._arg,
      $.splat_argument,
      $.hash_splat_argument,
      $.block_argument,
      $.pair
    ),

    splat_argument: $ => seq(alias($._splat_star, '*'), $._arg),
    hash_splat_argument: $ => seq('**', $._arg),
    block_argument: $ => seq(alias($._block_ampersand, '&'), $._arg),

    do_block: $ => seq(
      'do',
      optional($._terminator),
      optional(seq($.block_parameters, optional($._terminator))),
      $._body_statement
    ),

    block: $ => prec(PREC.CURLY_BLOCK, seq(
      '{',
      optional($.block_parameters),
      optional($._statements),
      '}'
    )),

    assignment: $ => prec.right(PREC.ASSIGN, choice(
      seq(
        field('left', choice($._lhs, $.left_assignment_list)),
        '=',
        field('right', choice(
          $._arg,
          $.splat_argument,
          $.right_assignment_list
        ))
      )
    )),

    command_assignment: $ => prec.right(PREC.ASSIGN, choice(
      seq(
        field('left', choice($._lhs, $.left_assignment_list)),
        '=',
        field('right', $._expression)
      )
    )),

    operator_assignment: $ => prec.right(PREC.ASSIGN, seq(
      field('left', $._lhs),
      choice('+=', '-=', '*=', '**=', '/=', '||=', '|=', '&&=', '&=', '%=', '>>=', '<<=', '^='),
      field('right', $._arg)
    )),

    command_operator_assignment: $ => prec.right(PREC.ASSIGN, seq(
      field('left', $._lhs),
      choice('+=', '-=', '*=', '**=', '/=', '||=', '|=', '&&=', '&=', '%=', '>>=', '<<=', '^='),
      field('right', $._expression)
    )),

    conditional: $ => prec.right(PREC.CONDITIONAL, seq(
      field('condition', $._arg),
      '?',
      field('consequence', $._arg),
      ':',
      field('alternative', $._arg)
    )),

    range: $ => prec.right(PREC.RANGE, seq($._arg, choice('..', '...'), $._arg)),

    binary: $ => {
      const operators = [
        [prec.left, PREC.AND, 'and'],
        [prec.left, PREC.OR, 'or'],
        [prec.left, PREC.BOOLEAN_OR, '||'],
        [prec.left, PREC.BOOLEAN_OR, '&&'],
        [prec.left, PREC.SHIFT, choice('<<', '>>')],
        [prec.left, PREC.COMPARISON, choice('<', '<=', '>', '>=')],
        [prec.left, PREC.BITWISE_AND, '&'],
        [prec.left, PREC.BITWISE_OR, choice('^', '|')],
        [prec.left, PREC.ADDITIVE, choice('+', alias($._binary_minus, '-'))],
        [prec.left, PREC.MULTIPLICATIVE, choice('/', '%', alias($._binary_star, '*'))],
        [prec.right, PREC.RELATIONAL, choice('==', '!=', '===', '<=>', '=~', '!~')],
        [prec.right, PREC.EXPONENTIAL, '**'],
      ];

      return choice(...operators.map(([fn, precedence, operator]) => fn(precedence, seq(
        field('left', $._arg),
        field('operator', operator),
        field('right', $._arg)
      ))));
    },

    command_binary: $ => prec.left(seq(
      field('left', $._expression),
      field('operator', choice('or', 'and')),
      field('right', $._expression)
    )),

    unary: $ => choice(
      prec(PREC.DEFINED, seq('defined?', $._arg)),
      prec.right(PREC.NOT, seq('not', $._arg)),
      prec.right(PREC.UNARY_MINUS, seq(choice(alias($._unary_minus, '-'), '+'), $._arg)),
      prec.right(PREC.COMPLEMENT, seq(choice('!', '~'), $._arg))
    ),

    parenthesized_unary: $ => prec(PREC.CALL, seq(
      choice('defined?', 'not'),
      $.parenthesized_statements
    )),

    unary_literal: $ => prec.right(PREC.UNARY_MINUS, seq(
      choice(alias($._unary_minus, '-'), '+'),
      choice($.integer, $.float)
    )),

    right_assignment_list: $ => prec(-1, commaSep1(choice($._arg, $.splat_argument))),

    left_assignment_list: $ => $._mlhs,
    _mlhs: $ => prec.left(-1, seq(
      commaSep1(choice($._lhs, $.rest_assignment, $.destructured_left_assignment)),
      optional(',')
    )),
    destructured_left_assignment: $ => prec(-1, seq('(', $._mlhs, ')')),

    rest_assignment: $ => prec(-1, seq('*', optional($._lhs))),

    _lhs: $ => prec.left(choice(
      $._variable,
      $.true,
      $.false,
      $.nil,
      $.scope_resolution,
      $.element_reference,
      $.call,
      $.method_call
    )),

    _variable: $ => prec.right(choice(
      $.self,
      $.super,
      $.instance_variable,
      $.class_variable,
      $.global_variable,
      $.identifier,
      $.constant
    )),

    operator: $ => choice(
      '..', '|', '^', '&', '<=>', '==', '===', '=~', '>', '>=', '<', '<=', '+',
      '-', '*', '/', '%', '!', '!~', '**', '<<', '>>', '~', '+@', '-@', '[]', '[]=', '`'
    ),

    _method_name: $ => choice(
      $.identifier,
      $.constant,
      $.setter,
      $.symbol,
      $.operator,
      $.instance_variable,
      $.class_variable,
      $.global_variable
    ),
    setter: $ => seq($.identifier, '='),

    undef: $ => seq('undef', commaSep1($._method_name)),
    alias: $ => seq(
      'alias',
      field('name', $._method_name),
      field('alias', $._method_name)
    ),

    comment: $ => token(prec(PREC.COMMENT, choice(
      seq('#', /.*/),
      seq(
        /=begin.*\r?\n/,
        repeat(choice(
          /[^=]/,
          /=[^e]/,
          /=e[^n]/,
          /=en[^d]/
        )),
        /=end/
      )
    ))),

    integer: $ => /0[bB][01](_?[01])*|0[oO]?[0-7](_?[0-7])*|(0[dD])?\d(_?\d)*|0x[0-9a-fA-F](_?[0-9a-fA-F])*/,

    float: $ => /\d(_?\d)*(\.\d)?(_?\d)*([eE][\+-]?\d(_?\d)*)?/,
    complex: $ => /(\d+)?(\+|-)?(\d+)i/,
    rational: $ => seq($.integer, 'r'),
    super: $ => 'super',
    self: $ => 'self',
    true: $ => token(choice('true', 'TRUE')),
    false: $ => token(choice('false', 'FALSE')),
    nil: $ => token(choice('nil', 'NIL')),

    constant: $ => token(seq(/[A-Z]/, IDENTIFIER_CHARS, /(\?|\!)?/)),
    identifier: $ => token(seq(LOWER_ALPHA_CHAR, IDENTIFIER_CHARS, /(\?|\!)?/)),
    instance_variable: $ => token(seq('@', ALPHA_CHAR, IDENTIFIER_CHARS)),
    class_variable: $ => token(seq('@@', ALPHA_CHAR, IDENTIFIER_CHARS)),

    global_variable: $ => /\$-?(([!@&`'+~=/\\,;.<>*$?:"])|([0-9]*)|([a-zA-Z_][a-zA-Z0-9_]*))/,

    chained_string: $ => seq($.string, repeat1($.string)),

    character: $ => /\?(\\\S({[0-9]*}|[0-9]*|-\S([MC]-\S)?)?|\S)/,

    interpolation: $ => seq(
      '#{', $._statement, '}'
    ),

    string: $ => seq(
      alias($._string_start, '"'),
      optional($._literal_contents),
      alias($._string_end, '"')
    ),

    subshell: $ => seq(
      alias($._subshell_start, '`'),
      optional($._literal_contents),
      alias($._string_end, '`')
    ),

    string_array: $ => seq(
      alias($._string_array_start, '%w('),
      optional(/\s+/),
      sep(alias($._literal_contents, $.bare_string), /\s+/),
      optional(/\s+/),
      alias($._string_end, ')')
    ),

    symbol_array: $ => seq(
      alias($._symbol_array_start, '%i('),
      optional(/\s+/),
      sep(alias($._literal_contents, $.bare_symbol), /\s+/),
      optional(/\s+/),
      alias($._string_end, ')')
    ),

    symbol: $ => choice($._simple_symbol, seq(
      alias($._symbol_start, ':"'),
      optional($._literal_contents),
      alias($._string_end, '"')
    )),

    regex: $ => seq(
      alias($._regex_start, '/'),
      optional($._literal_contents),
      alias($._string_end, '/')
    ),

    heredoc_body: $ => seq(
      $._heredoc_body_start,
      repeat(choice(
        $._heredoc_content,
        $.interpolation,
        $.escape_sequence
      )),
      $.heredoc_end
    ),

    _literal_contents: $ => repeat1(choice(
      $._string_content,
      $.interpolation,
      $.escape_sequence
    )),

    // https://ruby-doc.org/core-2.5.0/doc/syntax/literals_rdoc.html#label-Strings
    escape_sequence: $ => token(seq(
      '\\',
      choice(
        /[^ux0-7]/,          // single character
        /x[0-9a-fA-F]{1,2}/, // hex code
        /[0-7]{1,3}/,        // octal
        /u[0-9a-fA-F]{4}/,   // single unicode
        /u{[0-9a-fA-F ]+}/,  // multiple unicode
      )
    )),

    array: $ => seq(
      '[',
      optional($._argument_list_with_trailing_comma),
      ']'
    ),

    hash: $ => seq(
      '{',
      optional(seq(
        commaSep1(choice($.pair, $.hash_splat_argument)),
        optional(',')
      )),
      '}'
    ),

    pair: $ => choice(
      seq(
        field('key', $._arg),
        '=>',
        field('value', $._arg)
      ),
      seq(
        field('key', choice(
          alias($._identifier_hash_key, $.symbol),
          alias($.identifier, $.symbol),
          alias($.constant, $.symbol),
          $.string
        )),
        token.immediate(':'),
        field('value', $._arg)
      )
    ),

    lambda: $ => seq(
      '->',
      field('parameters', optional(choice(
        alias($.parameters, $.lambda_parameters),
        alias($.bare_parameters, $.lambda_parameters)
      ))),
      field('body', choice($.block, $.do_block))
    ),

    empty_statement: $ => prec(-1, ';'),

    _terminator: $ => choice(
      $._line_break,
      ';'
    ),
  }
});

function sep (rule, separator) {
  return optional(sep1(rule, separator));
}

function sep1 (rule, separator) {
  return seq(rule, repeat(seq(separator, rule)));
}

function commaSep1 (rule) {
  return sep1(rule, ',');
}

function commaSep (rule) {
  return optional(commaSep1(rule));
}