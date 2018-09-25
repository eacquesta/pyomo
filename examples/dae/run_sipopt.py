# ___________________________________________________________________________
#    
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and 
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License.
# ___________________________________________________________________________

# Author: Erin Acquesta, 2018-02-14
# 
# This pyomo function is designed to accept the following inputs
#   m : pyomo model object
#   p : list of parameters
#   eps : list of values to perturbed p by. 
#           Note: length(eps) must equal length(p)  
# 


from pyomo.environ import *
from pyomo.core.base import _ConstraintData, _ObjectiveData, _ExpressionData
from pyomo.core.expr.current import (clone_expression, identify_variables, 
                                     ExpressionReplacementVisitor)
from pyomo.common.modeling import unique_component_name
from pyomo.opt import SolverFactory

from HIV_Transmission_discrete import m


#------------------------------------------#
#	User Inputs
#------------------------------------------#
paramSubList = [m.aa,m.eps,m.qq]
perturbList = [m.aaDelta, m.epsDelta, m.qqDelta]

paramPerturbMap = ComponentMap(zip(paramSubList,perturbList))

#Expand parameter component
perturbSubMap = {}
paramFullList = []
#varSubList = []
for parameter in paramSubList:
    #Loop over each ParamData in the paramter (will this work on sparse params?)
    for kk in parameter:
        perturbSubMap[id(parameter[kk])]=paramPerturbMap[parameter][kk]
        paramFullList.append(parameter[kk])

#------------------------------------------#
#       Model Translations
#------------------------------------------#

m.b=Block()

varSubList = []
#add variable components for identified parameters
#Parameters must be mutable
for parameter in paramSubList:
    tempName = unique_component_name(m,parameter.local_name) 
#    m.b.add_component(tempName,Var(parameter._data.keys()))
    m.b.add_component(tempName,Var(parameter.index_set()))
    myVar = m.b.component(tempName)
    varSubList.append(myVar)

#----------------------------------------------------------#
# Should we consider cloning the whole model? 
#	- current code deactivates user's Objective
#	 and Constraints. 
#	- substitutions with Expressions are problematic
#	because it would require another type of 
#	substitution.
#	- If we clone the whole model we can muck up 
#	whatever we want to and leave the user's model in
#	original form.
# Maybe set an option, then the user can decide.
#----------------------------------------------------------#


#Param substitution map
#varSubList = [m.b.aa_var, m.b.eps_var, m.b.qq_var]
paramCompMap = ComponentMap(zip(paramSubList, varSubList))

#Loop through components to build substitution map
variableSubMap = {}
#paramFullList = []
for parameter in paramSubList:
    #Loop over each ParamData in the paramter (will this work on sparse params?)
    for kk in parameter:
        variableSubMap[id(parameter[kk])]=paramCompMap[parameter][kk]
        #variableSubMap[id(pp)] = paramCompMap[pp][kk]
#        paramFullList.append(parameter[kk])

#------------------------------------------i----------------#
#Objective, Expression, and Constraints will ALL be cloned
#	regardless if needed.
#	Need to consider an efficient way to handle cloning
#	only when needed.
#----------------------------------------------------------#

#substitute the Objectives
for cc in list(m.component_data_objects(Objective,
                                        active=True,
                                        descend_into=True)):
    tempName=unique_component_name(m,cc.local_name)    
    m.b.add_component(tempName,
                  Objective(expr=ExpressionReplacementVisitor(
                  substitute=variableSubMap,
                  remove_named_expressions=True).dfs_postorder_stack(cc.expr)))
    cc.deactivate()

    #-------------------------------------------------------------------#
    #Can NOT Deactive Expressions. Need to consider how to handle model 
    #	Expression calls. We need a substitution mechanism or work
    #   around to address the existance fo the original Expression
    #   as well as the cloned Expression. 
    #-------------------------------------------------------------------#

#substitue the Constraints while using a constraint list
#m.b.conlist=ConstraintList()
    #---------------------------------------#
    #  Currenlty not using the constraint
    #  list for constraint substitutions
    #  *****should ask about this*****
    #---------------------------------------#

m.b.constList = ConstraintList()
for cc in list(m.component_data_objects(Constraint, 
                                   active=True,
                                   descend_into=True)):
    if cc.equality:
        m.b.constList.add(expr= ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
#        m.b.add_component(cc.local_name,
#                          Constraint(expr=
#                                     clone_expression(cc.expr,
#                                                   substitute=variableSubMap)))
    else:
        if cc.lower==None or cc.upper==None:
            m.b.constList.add(expr=ExpresssionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
#            m.b.add_component(cc.local_name,
#                              Constraint(expr=
#                                         clone_expression(cc.expr,
#                                                   substitute=variableSubMap)))
        else:
            m.b.constList.add(expr=ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr)
                <=ExpressionReplacementVisitor(
                    substitute=variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr)
                <=ExpressionReplacementVisitor(
                    substitute_variableSubMap,
                    remove_named_expressions=True).dfs_postorder_stack(cc.expr))
#            m.b.add_component(cc.local_name,
#                              Constraint(expr=
#                                         clone_expression(cc.lower,
#                                                   substitute=variableSubMap)
#                                         <=clone_expression(cc.body,
#                                                   substitute=variableSubMap)
#                                         <=clone_expression(cc.upper,
#                                                   substitute_variableSubMap)))
#        
    cc.deactivate()


#-----------------------------------------------------#
#
#  paramData to varData constraint list
#  
#-----------------------------------------------------#

m.b.paramConst = ConstraintList()
for ii in paramFullList:
    jj=variableSubMap[id(ii)]
    m.b.paramConst.add(ii==jj)

    
#----------------------------------------#
#           sIPOPT
#----------------------------------------#
    
#Create the ipopt_sens (aka sIPOPT) solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = True    #True prints solver output to screen
keepfiles = False   #True prints intermediate file names (.nl, .sol, ....)
opt = SolverFactory(solver, solver_io=solver_io)

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for 'ipopt_sens' ")
    print("")
    exit(1)
    
#Declare Suffixes
m.sens_state_0 = Suffix(direction=Suffix.EXPORT)
m.sens_state_1 = Suffix(direction=Suffix.EXPORT)
m.sens_state_value_1 = Suffix(direction=Suffix.EXPORT)
m.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT)
m.sens_init_constr = Suffix(direction=Suffix.EXPORT)


#set sIPOPT data
opt.options['run_sens'] = 'yes'

kk=1
for ii in paramFullList:
    m.sens_state_0[variableSubMap[id(ii)]] = kk
    m.sens_state_1[variableSubMap[id(ii)]] = kk
    m.sens_state_value_1[variableSubMap[id(ii)]] = value(perturbSubMap[id(ii)])
    m.sens_init_constr[m.b.paramConst[kk]] = kk
    kk += 1    

#Send the model to the ipopt_sens and collect the solution
results = opt.solve(m, keepfiles=keepfiles, tee=stream_solver)
#
#Print solution
print("Nominal and perturbed solution:")
for vv in [varSubList[1],m.L[m.tf]]:
    print("%5s %14g %14g" % (vv, value(vv), m.sens_sol_state_1[vv]))
###
##
#
# return m.sens_sol_state_1