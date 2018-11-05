library("igraph")

# configuration
colormap = c('grey', 'green', 'red', 'blue') # 'absorbed' beat colored by blue, 
                                             # 'radiative' beat colored by red,
                                             # 'equal' beat colored by green,
                                             # 'isolated' beat colored by grey.

# load data
cor.df  = read.table('/Users/woodie/Desktop/workspace/Event-Series-Detection/result/init-Amatrix.txt', sep=',', header=FALSE)
cor.mat = data.matrix(cor.df)
cor.mat = cor.mat[-1, -1]
cor.mat = cor.mat[-dim(cor.mat)[1], -dim(cor.mat)[2]]
diag(cor.mat) = 0 # set diagonal of matrix to be zero (ignore the influence by itself)

# beat labels
burglary.beats = c('101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '501', '502', '504', '505', '506', '507', '508', '509', '510', '511', '512', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '612')
robbery.beats  = c('101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '201', '202', '203', '204', '206', '207', '208', '209', '210', '211', '212', '213', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '401', '402', '403', '404', '405', '407', '408', '409', '410', '411', '412', '413', '414', '501', '502', '503', '504', '505', '506', '507', '508', '509', '511', '512', '601', '602', '603', '604', '605', '606', '608', '609', '610', '612')
other.beats    = c('101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612')

# get adjacency matrix by eliminating edges with small coefficients
threshold = 0.5
adj.mat   = cor.mat >= threshold
sprintf('Number of non-zero entries in adjacent matrix is %d', sum(adj.mat>0))

# calculating each beat 'absorbed' or 'radiative'.
num.absorbed  = as.vector(rowSums(adj.mat))
num.radiative = as.vector(colSums(adj.mat))
num.direct    = num.absorbed - num.radiative
num.edge      = num.absorbed + num.radiative
# set color according to the state of beats
beats.df       = data.frame(num.direct, num.edge)
beats.df$state = as.integer(num.edge > 0)
beats.df[beats.df$num.edge > 0 & beats.df$num.direct == 0, 'state'] = 1
beats.df[beats.df$num.direct > 0, 'state'] = 2
beats.df[beats.df$num.direct < 0, 'state'] = 3
beats.df$state = beats.df$state + 1
size.beats  = beats.df$num.edge/3
color.beats = colormap[beats.df$state]

g <- igraph::graph.adjacency(adj.mat, mode="directed")
plot.igraph(g, 
            vertex.label=burglary.beats,
            vertex.label.color="black",
            vertex.label.family="Times", # Font family of the label (e.g.“Times”, “Helvetica”)
            vertex.label.font=3,         # Font: 1 plain, 2 bold, 3, italic, 4 bold italic, 5 symbol
            vertex.label.cex=.8,         # Font size (multiplication factor, device-dependent)
            vertex.label.dist=.5,        # Distance between the label and the vertex
            # vertex.label.degree=0 ,      # The position of the label in relation to the vertex (use pi)
            vertex.size=size.beats,
            vertex.color=color.beats,
            vertex.shape="circle",    # One of “none”, “circle”, “square”, “csquare”, “rectangle” “crectangle”, “vrectangle”, “pie”, “raster”, or “sphere”
            vertex.size=5,            # Size of the node (default is 15)
            edge.arrow.size=0.3,      # Arrow size, defaults to 1
            edge.arrow.width=1)

# convert influential matrix into a symmetrical positive definite matrix
# (undirected graph)
# near.pd = nearPD(forceSymmetric(cor.mat, 'L'))
# weights = near.pd$mat