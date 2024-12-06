export interface SubFooter {
  copywriteText: string;
}

export interface FooterData {
  subFooter: SubFooter;
}

export const footerNavigationData: FooterData = {
  subFooter: {
    copywriteText: "Splinter 2024.",
  },
};
