import ExperimentTrackingRoutes from '../../experiment-tracking/routes';
import { Link, Location, matchPath, useLocation } from '../utils/RoutingUtils';
import logo from '../../common/static/RapidFire_Square_Bug.png';
import { ModelRegistryRoutes } from '../../model-registry/routes';

import { useQueryClient } from '@tanstack/react-query';
import { Typography, Button } from '@databricks/design-system';
import { useState, useEffect } from 'react';

const colors = {
  headerBg: '#0b3574',
  headerText: '#e7f1fb',
  headerActiveLink: '#43C9ED',
};

const classNames = {
  activeNavLink: { borderBottom: `4px solid ${colors.headerActiveLink}` },
};

const isModelsActive = (location: Location) => matchPath('/models/*', location.pathname);

// Update the isExperimentsActive function to account for files
const isExperimentsActive = (location: Location) => !isModelsActive(location);

export const MlflowHeader = ({
  isDarkTheme = false,
  setIsDarkTheme = (val: boolean) => {},
}: {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}) => {
  const location = useLocation();
  const HEADER_HEIGHT = '70px';

  return (
    <>
      <header
        css={{
          backgroundColor: colors.headerBg,
          height: HEADER_HEIGHT,
          color: colors.headerText,
          display: 'flex',
          gap: 24,
          a: {
            color: colors.headerText,
          },
        }}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'flex-end',
          }}
        >
          <Link to={ExperimentTrackingRoutes.experimentsObservatoryRoute}>
            <img
              css={{
                height: 45,              
                marginLeft: 24,
                marginTop: 12,           
                marginBottom: 12,       
              }}
              alt="MLflow"
              src={logo}
            />
          </Link>
        </div>
        <div
          css={{
            display: 'flex',
            paddingTop: 25, 
            fontSize: 18,  
            gap: 24,
            '& a': {      
              fontWeight: 500,  
              transition: 'color 0.2s ease',  
              '&:hover': {
                color: colors.headerActiveLink,  
              }
            }
          }}
        >
          <Link
            to={ExperimentTrackingRoutes.experimentPageDefaultRoute}
            style={isExperimentsActive(location) ? classNames.activeNavLink : undefined}
          >
            Experiments
          </Link>
          {/* <Link
            to={ModelRegistryRoutes.modelListPageRoute}
            style={isModelsActive(location) ? classNames.activeNavLink : undefined}
          >
            Models
          </Link> */}
        </div>
        <div css={{ flex: 1 }} />
        
        <div 
          css={{ 
            display: 'flex', 
            gap: 24, 
            paddingTop: 12,
            fontSize: 16, 
            marginRight: 24,
            alignItems: 'center'
          }}
        >
          <a
            href="https://rapidfire-ai-oss-docs.readthedocs-hosted.com/en/latest/"
            target="_blank"
            rel="noopener noreferrer"
            css={{
              color: colors.headerText,
              textDecoration: 'none',
              fontSize: 18,  
              fontWeight: 500,
              transition: 'color 0.2s ease',
              '&:hover': {
                color: colors.headerActiveLink,
              }
            }}
          >
            Docs
          </a>
        </div>
      </header>
    </>
  );
};
