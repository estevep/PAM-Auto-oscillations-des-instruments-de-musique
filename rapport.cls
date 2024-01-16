%-------------------------------------------------------------------------------
%                identification
%-------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rapport}[16/02/2023 v1.0.1 ma classe de rapport]

%-------------------------------------------------------------------------------
%                Required packages
%-------------------------------------------------------------------------------
\RequirePackage [T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[german,english,french]{babel}
%! order of languages affects language used for chapters

\RequirePackage{xkeyval}
%-------------------------------------------------------------------------------
%                Options
%-------------------------------------------------------------------------------
% Options variables
%---------
% defines the document logo
% usage: \logo{<logo_path>}
\def\logo#1{\def\@logo{#1}}

% Book indicator
\newif\if@book

%---------
% Declare options
%---------
\DeclareOption{phelma_black}{
  \logo{Grenoble INP - Phelma (noir, RVB)}
}
\DeclareOption{phelma_normal}{
  \logo{Grenoble INP - Phelma (couleur, RVB)}
}
\DeclareOption{phelma_old}{
  \logo{logo_phelma}
}
\DeclareOption{kit_de}{
  \logo{kitlogo_de_rgb}
}
\DeclareOption{kit_en}{
  \logo{kitlogo_en_rgb}
}
\DeclareOption{atiam}{
  \logo{ATIAM_logo_LONG_Q}
}


\DeclareOption{book}{
  \gdef\chosenclass{book}  
  \@booktrue
}
\DeclareOption{article}{
  \gdef\chosenclass{article}
  \@bookfalse
}


% execute default options
% \ExecuteOptions{phelma_normal}
\ExecuteOptions{phelma_normal, article}

%---------
% Process given options
%---------
\ProcessOptions\relax

\LoadClass{\chosenclass}

%-------------------------------------------------------------------------------
%                Required packages
%-------------------------------------------------------------------------------

\RequirePackage{fullpage}

%\RequirePackage{tcolorbox} % pour les boites des CR d'élec

%---------
% Inkscape figures in LaTeX -> see \incfig command in command section
%---------

% Pour ajouter des pdf_tex générés par inkscape (mais ça bug pour l'instant)
\RequirePackage{import}
\RequirePackage{xifthen}
% \RequirePackage{pdfpages} % Already included for broader use
\RequirePackage{transparent}

% ------------

\RequirePackage[table,xcdraw]{xcolor}%               pour gerer les couleurs
\RequirePackage{rotating} % rotate any object of an arbitrary angle

\RequirePackage[european, siunitx]{circuitikz} % circuits électriques
\RequirePackage{siunitx}

\sisetup{
    detect-all,
    output-decimal-marker={,},
    group-minimum-digits = 3,
    group-separator={~},
    number-unit-separator={~},
    inter-unit-product={~},
    % math-celsius = °\text{C}, % for temperatures
    % text-celsius = °C,
    % math-degree = °, % for angles
    % text-degree = °
}
\usetikzlibrary{arrows.meta,quotes,positioning,babel}
\def\distance{0.5}
\def\leftColor{red}
\def\rightColor{blue}
\def\Top{4}
\def\Bottom{1}
\tikzset{
   sign1/.style={anchor=center, text=\leftColor},
   sign2/.style={anchor=center, text=\rightColor},
}


% \RequirePackage{tikz} % dessin scientifique
\RequirePackage[justification=centering]{caption} % titre des figures
\RequirePackage{subcaption} % subfigures
\RequirePackage{cprotect} % verb in captions
\RequirePackage{graphicx} % améliore le includegraphics
%\RequirePackage{floatrow}  % pour les figures
%\RequirePackage{float}  % pour les figures
\RequirePackage{gensymb} % apporte \degree, \celsius, \perthousand, \micro and \ohm which work both in text and maths mode

\setlength{\parindent}{0cm} % Indentation à chaque paragraphe
\setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}
\newcommand{\hsp}{\hspace{20pt}}
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

\RequirePackage{mathtools}
\RequirePackage{diagbox}%              barres diagonales dans les tableaux
\RequirePackage{textcomp}%             caracteres additionnels
\RequirePackage{amsmath,amssymb}%      pour les maths

\RequirePackage{diffcoeff}

\RequirePackage{lmodern}%              variantes : txfonts, fourier, etc.
\RequirePackage[a4paper]{geometry}%    taille correcte du papier, % showframe option to show the frame
\RequirePackage{microtype}%            ameliorations typographiques
\RequirePackage{listings}%             pour le code informatique
\RequirePackage{varioref} % références lointaines
\RequirePackage[pdftex, unicode]{hyperref}%     gestion des hyperliens
\RequirePackage{cleveref}%             références à côté
\hypersetup{pdfstartview=XYZ}%     zoom par defaut
\RequirePackage{xspace}%			  guillemets
\RequirePackage{tabto}%				  tabulations
\RequirePackage{verbatim}
\RequirePackage{fancyhdr}%              Joli en-tête et pied de page
\RequirePackage{stmaryrd}%             brackets
\RequirePackage{float}
\RequirePackage{aliascnt}%				equations caption

\RequirePackage{pdfpages}%            include pdf files inside document

\RequirePackage{contour}%           in order to create a nice underline but basically creates a contour of a given color around the letters of the given words \contour{color}{contoured text}
\RequirePackage[normalem]{ulem}%            gives access to uline without changing \emph{}

%---------
% Bibliography
%---------
\RequirePackage{biblatex}[
    backend=biber,        % compilateur par défaut pour biblatex
    sorting=nyt,          % trier par nom, année, titre
    citestyle=authoryear, % style de citation auteur-année
    bibstyle=alphabetic,  % style de bibliographie alphabétique
]
\RequirePackage{csquotes}



% \AtEndPreamble hook (loading etoolbox instead of defining the macro, as to avoid incompatibilities with etoolbox (and packages relying on it) defining the macro too)
\RequirePackage{etoolbox}

%\RequirePackage{fourier}
% \epstopdfsetup{outdir=./}
\graphicspath{ {Images/} }



%-------------------------------------------------------------------------------
%                Page structure
%-------------------------------------------------------------------------------

% \setlength{\parskip}{1ex plus 0.5ex minus 0.2ex}
% \setlength{\hoffset}{-35pt}
% \setlength{\textwidth}{500pt} % Largeur de la zone de texte
% \setlength{\voffset}{-18pt} % Bon pour DOS
% % \setlength{\marginparsep}{7pt} % Séparation de la marge
% \setlength{\topmargin}{0pt} % Pas de marge en haut
% \setlength{\headheight}{35pt} % Haut de page
% \setlength{\headsep}{10pt} % Entre le haut de page et le texte
% \setlength{\footskip}{40pt} % Bas de page + séparation
% \setlength{\textheight}{660pt} % Hauteur de la zone de texte (25cm)

\geometry{
  a4paper,
  left=2cm,
  right=2cm,
  top=1cm,
  bottom=4cm
}

\headheight 40pt              %% put this outside
\headsep 10pt                 %% put this outside

\pagestyle{fancy}

\renewcommand{\footrulewidth}{1pt}
\fancyfoot[L]{Page n°\thepage}
\fancyfoot[c]{\@author}
\fancyfoot[R]{\@serie}

\fancyhead[R]{
  \ifthenelse {\isundefined{\@titleheader}} {
      \ifthenelse {\isundefined{\@title}} {~} {\@title}
  }
  {\@titleheader}
}
% \fancyhead[L]{\includegraphics[scale=0.04]{Images/logo_phelma.png}}%Images/
\fancyhead[L]{\includegraphics[height=1.1cm]{logo/\@logo}}%Images/

\if@book
  \let\ps@plain\ps@fancy
\fi
% Chapter pages use the plain pagestyle -> make plain=fancy so that the chapter pages also include the headers


% \newcommand*{\doctitle}[1]{\def\@doctitle{#1}}
\AtEndPreamble{
  \hypersetup{ % paramètres des liens hypertextes
      pdftitle = {\@doctitle{}},
      colorlinks=true,
      linkcolor=blue!65!green,
      citecolor=bleu_cite,
  }
}
% \AtEndPreamble{}

%---------
% Title page
%---------

\renewcommand{\maketitle}{

  \begin{titlepage}
    % \begin{sffamily}
    \begin{center}

      % Upper part of the page. The '~' is needed because \\
      % only works if a paragraph has started.

      % \includegraphics[scale=0.15]{Images/logo_phelma.png}~\\[1.5cm] %Images/
      \includegraphics[height=3.8cm]{logo/\@logo}~\\[1.5cm] %Images/

      \textsc{\LARGE \ifthenelse {\equal{\@titleone}{}} {~} {\@titleone} }\\[1cm]%[.5cm] if another logo has to be added

      \textsc{\huge \ifthenelse {\equal{\@titletwo}{}} {~} {\@titletwo}}\\[1cm]
   	\textsc{\Large \ifthenelse {\equal{\@titlethree}{}} {~} {\@titlethree}}
    \\[0.5cm]
  	{\color{black}

      	% Title
      	\HRule \\[0.4cm]
      	{ \Huge \bfseries \@title\\[0.4cm] }

      	\HRule \\[2cm]
      	}

        % Add other logos here if needed with includegraphics and hfill



  	\vfill{

      % Author and supervisor
      \begin{minipage}{0.4\textwidth}
        \begin{flushleft} \Large
          \ifthenelse {\isundefined{\@autpage}} {
              \ifthenelse {\isundefined{\@author}} {~} {\@author}
          }
          {\@autpage}
        \end{flushleft}
      \end{minipage}
      \begin{minipage}{0.5\textwidth}
        \begin{flushright} \Large
        % \@supervisor

        \ifthenelse{\isundefined{\@supervisorMail}}{
          \@supervisor
        }
        {
          \@supervisor\\
          \emailsymbol \@supervisorMail\\%[1cm]
        }
        \end{flushright}
      \end{minipage}
      \begin{minipage}{0.4\textwidth}
      	\begin{center} \Large
      		~\\~\\\emph\@serie\\
      	\end{center}
      \end{minipage}
      ~\\~\\~\\~\\
      % }
  	% \vfill
    %   Bottom of the page
      \large \@date}

    \end{center}
   % \end{sffamily}
  \end{titlepage}
}


%-------------------------------------------------------------------------------
%                Styles
%-------------------------------------------------------------------------------

%---------
% Colors
%---------
\definecolor{mymauve}{rgb}{0.58,0,0.82}
\definecolor{mygreen}{rgb}{0,0.6,0}
\definecolor{mygray}{rgb}{0.5,0.5,0.5}
\definecolor{myred}{rgb}{1,0.18,0.14}
\definecolor{bleu_cite}{RGB}{12,127,172}
\definecolor{deepblue}{rgb}{0,0,0.5}
\definecolor{deepred}{rgb}{0.6,0,0}
\definecolor{deepgreen}{rgb}{0,0.5,0}

%---------
% Create nice underline
%---------

\renewcommand{\ULdepth}{1.8pt}
\contourlength{0.8pt}

\newcommand{\myuline}[1]{%
  \uline{\phantom{#1}}%
  \llap{\contour{white}{#1}}%
}

%---------
% Symbols
%---------

\newcommand{\Hsquare}{%
  \text{\fboxsep=-.2pt\fbox{\rule{0pt}{1ex}\rule{1ex}{0pt}}}%
}
%Panneau attention
\newcommand{\attention}{{\fontencoding{U}\fontfamily{futs}\selectfont\char 66\relax}}

%Flèche relative
\newcommand{\subarrow}{\qquad\rotatebox[origin=c]{180}{$\Lsh$}\ }

\RequirePackage{marvosym}

\newcommand*{\marvosymbol}[1]{{\fontfamily{mvs}\fontencoding{U}\fontseries{m}\fontshape{n}\selectfont\char#1}}
\newcommand*{\emailsymbol}         {\marvosymbol{66}~}

%-------------------------------------------------------------------------------
%                Coding environments
%-------------------------------------------------------------------------------

% enables to type latex code block in a listings environment
%
% usage:  \begin{code-latex}{line_number, numberstyle}
%           <code>
%         \end{code-latex}
%
% Parameters:
% line_number (optional) = start line enumeration at line_number
% numberstyle = where to put the line-numbers; possible values are (none, left, right)
\lstnewenvironment{code-latex}[1][]{
	\lstset{
		backgroundcolor=\color{green!5},   % choose the background color; you must add \RequirePackage{color} or \RequirePackage{xcolor}; should come as last argument
		upquote=true,
		columns=flexible,
		basicstyle=\ttfamily,
		language={[LaTeX]TeX},
		texcsstyle=*\color{blue},
		commentstyle=\color{gray},
		moretexcs={abslabeldelim,setlength,abstitleskip},#1
	}
}{}

% enables to type c code block in a listings environment
%
% usage:  \begin{code-c}{line_number, numberstyle}
%           <code>
%         \end{code-c}
%
% Parameters:
% line_number (optional) = start line enumeration at line_number
% numberstyle = where to put the line-numbers; possible values are (none, left, right)
\lstnewenvironment{code-c}[2][1]{
	\lstset{
		backgroundcolor=\color{green!5},   % choose the background color; you must add \RequirePackage{color} or \RequirePackage{xcolor}; should come as last argument
		upquote=true,
		columns=flexible,
		basicstyle=\ttfamily,
		language={C},
		keywordstyle=\bfseries\color{blue},
		commentstyle=\itshape\color{gray},
		%keywordstyle=\bfseries\color{green!40!black},
  		%identifierstyle=\color{mymauve},
		firstnumber=#1,                % start line enumeration with line 1000
		%breaklines=true,                 % sets automatic line breaking
  		numbers=#2,                    % where to put the line-numbers; possible values are (none, left, right)
		numberstyle=\small\color{mygray}, % the style that is used for the line-numbers
	}
}{}

% enables to type bash code block in a listings environment
%
% usage:  \begin{code-bash}{line_number, numberstyle}
%           <code>
%         \end{code-bash}
%
% Parameters:
% line_number (optional) = start line enumeration at line_number
% numberstyle = where to put the line-numbers; possible values are (none, left, right)
\lstnewenvironment{code-bash}[2][1]{
	\lstset{
		backgroundcolor=\color{green!5},   % choose the background color; you must add \RequirePackage{color} or \RequirePackage{xcolor}; should come as last argument
		upquote=true,
		columns=flexible,
		basicstyle=\ttfamily,
		language={bash},
		keywordstyle=\bfseries\color{blue},
		commentstyle=\itshape\color{gray},
		%keywordstyle=\bfseries\color{green!40!black},
  		%identifierstyle=\color{mymauve},
		firstnumber=#1,                % start line enumeration with line 1000
  		numbers=#2,                    % where to put the line-numbers; possible values are (none, left, right)
		%breaklines=true,                 % sets automatic line breaking
		numberstyle=\small\color{mygray}, % the style that is used for the line-numbers
		showstringspaces=false,          % underline spaces within strings only
		morekeywords={ls,mkdir,touch}
	}
}{}

% enables to type test code block in a listings environment # Projet MIPS
%
% usage:  \begin{code-test}{line_number, numberstyle}
%           <code>
%         \end{code-test}
%
% Parameters:
% line_number (optional) = start line enumeration at line_number
% numberstyle = where to put the line-numbers; possible values are (none, left, right)
\lstnewenvironment{code-test}[2][1]{
	\lstset{
		backgroundcolor=\color{green!5},   % choose the background color; you must add \RequirePackage{color} or \RequirePackage{xcolor}; should come as last argument
		upquote=true,
		columns=flexible,
		basicstyle=\ttfamily,
		language={bash},
		keywordstyle=\bfseries\color{blue},
		commentstyle=\itshape\color{gray},
		%keywordstyle=\bfseries\color{green!40!black},
  		%identifierstyle=\color{mymauve},
		firstnumber=#1,                % start line enumeration with line 1000
  		numbers=#2,                    % where to put the line-numbers; possible values are (none, left, right)
		%breaklines=true,                 % sets automatic line breaking
		numberstyle=\small\color{mygray}, % the style that is used for the line-numbers
		showstringspaces=false,          % underline spaces within strings only
		morekeywords={assert, set, disp, debug, resume, load, exit}
	}
}{}

% Python style for highlighting
\newcommand*{\pythonstyle}{
  \lstset{
    backgroundcolor=\color{green!5},   % choose the background color; you must add \RequirePackage{color} or \RequirePackage{xcolor}; should come as last argument
    upquote=true,
    columns=flexible,
    basicstyle=\ttfamily,
    language={python},
    keywordstyle=\bfseries\color{blue},
    commentstyle=\itshape\color{gray},
    emph={shape,__init__},          % Custom highlighting
    emphstyle=\bfseries\color{deepred},    % Custom highlighting style
    stringstyle=\color{deepgreen},
    %keywordstyle=\bfseries\color{green!40!black},
      %identifierstyle=\color{mymauve},
    breaklines=true,                 % sets automatic line breaking
    numberstyle=\small\color{mygray}, % the style that is used for the line-numbers
    showstringspaces=false,          % underline spaces within strings only
    morekeywords={self, rectangle, opening, reconstruction, zeros}
    % frame=tb,                         % Any extra options here, frame = tb adds a frame
  }
}


% enables to type python code block in a listings environment
%
% usage:  \begin{code-python}{line_number, numberstyle}
%           <code>
%         \end{code-python}
%
% Parameters:
% line_number (optional) = start line enumeration at line_number
% numberstyle = where to put the line-numbers; possible values are (none, left, right)
\lstnewenvironment{code-python}[2][1]{
  \pythonstyle
  \lstset{
		firstnumber=#1,                % start line enumeration with line 1000
  	numbers=#2,                    % where to put the line-numbers; possible values are (none, left, right)
		% breaklines=true,                 % sets automatic line breaking
	}
}{}


% Python environment
% enables to type python code block in a listings environment
% usage:  \begin{python}{options}
%           <code>
%         \end{python}
\lstnewenvironment{python}[1][]{
    \pythonstyle
    \lstset{#1}
}{}

% Python for external files
% enables to include python files
% usage: \pythonexternal{filename.py}
\newcommand*{\pythonexternal}[2][]{{
\pythonstyle
\lstinputlisting[#1]{#2}}}

% Python for inline
% enables to type inline code
% usage: \pythoninline{<code>}
\newcommand*{\pythoninline}[1]{{\pythonstyle\lstinline!#1!}}


%-------------------------------------------------------------------------------
%                Other environments
%-------------------------------------------------------------------------------

\newaliascnt{eqfloat}{equation}
\newfloat{eqfloat}{h}{eqflts}
\floatname{eqfloat}{Equation}

\newcommand*{\ORGeqfloat}{}
\let\ORGeqfloat\eqfloat
\def\eqfloat{%
  \let\ORIGINALcaption\caption
  \def\caption{%
    \addtocounter{equation}{-1}%
    \ORIGINALcaption
  }%
  \ORGeqfloat
}

%-------------------------------------------------------------------------------
%                overall design commands definitions
%-------------------------------------------------------------------------------
% elements
%---------
% defines the document title
% usage: \doctitle{<document_title>}
\newcommand*{\doctitle}[1]{\def\@doctitle{#1}}

% defines the serie
% usage: \serie{<serie>}
\newcommand*{\serie}[1]{\def\@serie{#1}}

% defines the author for title page
% usage: \autpage{<author>}
\newcommand*{\autpage}[1]{\def\@autpage{#1}}

% defines the supervisor
% usage: \supervisor[<function>]{<name of the supervisor>}
\newcommand*{\supervisor}[2][1]{\def\@supervisor{\emph{\ifthenelse{\equal{#1}{1}}{}{#1}}#2}}

% defines the supervisor
% usage: \supervisor[<function>]{<name of the supervisor>}
\newcommand*{\supervisorMail}[1]{\def\@supervisorMail{#1}}

% defines the header title
% usage: \titleone{<titleone>}
\newcommand*{\titleheader}[1]{\def\@titleheader{#1}}

% defines the first title
% usage: \titleone{<titleone>}
\newcommand*{\titleone}[1]{\def\@titleone{#1}}

% defines the second title
% usage: \titletwo{<titletwo>}
\newcommand*{\titletwo}[1]{\def\@titletwo{#1}}

% defines the third title
% usage: \titlethree{<titlethree>}
\newcommand*{\titlethree}[1]{\def\@titlethree{#1}}

%---------
% sections
%---------

% \newcommand*{\myparagraph}[1]{
%   \paragraph{#1}
%   ~\newline
% }

\newcommand*\myparagraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {1.5ex \@plus .2ex}
                                    {\normalfont\normalsize\bfseries}}

%---------
% table of contents
%---------

% creates table of contents without colored links
% usage: \toc
\newcommand*{\toc}{
    {\hypersetup{hidelinks} \tableofcontents }
    \noindent\hrulefill
}

% creates table of contents without colored links on a full page
% usage: \tocpage
\newcommand*{\tocpage}{
    {\hypersetup{hidelinks} \tableofcontents }
    \newpage
}

% creates list of figures without colored links
% usage: \lof
\newcommand*{\lof}{
    {\hypersetup{hidelinks} \listoffigures }
    % \noindent\hrulefill
}

% creates list of tables without colored links
% usage: \lot
\newcommand*{\lot}{
    {\hypersetup{hidelinks} \listoftables }
    % \noindent\hrulefill
}

%---------
% Inkscape figures
%---------

\newcommand{\incfig}[1]{%
    \def\svgwidth{\columnwidth}
    \import{./figures/}{#1.pdf_tex}
}

%---------
% Math shortcuts
%---------

\newcommand*{\bg}[1]{\mathbf{#1}}
\newcommand*{\bb}[1]{\mathbb{#1}}

\newcommand*{\R}{\mathbb{R}}
\newcommand*{\N}{\mathbb{N}}
\newcommand*{\Z}{\mathbb{Z}}
\newcommand*{\Q}{\mathbb{Q}}
\newcommand*{\C}{\mathbb{C}}
\newcommand*{\K}{\mathbb{K}}

%----------------------------------------------------------------------------------------
